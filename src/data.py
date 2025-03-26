import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


def create_synthetic_cache_dir(base_dir: str, gan_path: str, rate: float) -> str:
    gan_hash = hashlib.md5(gan_path.encode()).hexdigest()[:8]
    rate_tag = f"rate{int(rate * 100)}"
    return str(Path(base_dir) / f"gan_{gan_hash}_{rate_tag}")


class FER2013CSV(Dataset):
    def __init__(self, csv_file: str, split: str = 'Training'):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['Usage'] == split]
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pixels = np.array(row['pixels'].split(), dtype=np.uint8)
        image = Image.fromarray(pixels.reshape(48, 48))
        label = int(row['emotion'])
        return self.transform(image), label


class _SyntheticDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = list(Path(image_dir).rglob('*.png'))
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        label = int(path.name.split('_')[-1].split('.')[0])
        return self.transform(img), label


class FERData(L.LightningDataModule):
    def __init__(
        self,
        generator,
        gan_path,
        data_dir,
        real_path=None,
        rate=0.0,
        total_num=28709,
        num_classes=7,
        batch_size=64,
        num_workers=4,
        force_refresh=False,
    ):
        super().__init__()
        self.val_dataset = None
        self.test_dataset = None
        self.train_dataset = None
        self.G = generator
        self.gan_path = gan_path
        self.real_path = real_path
        self.rate = rate
        self.total_num = total_num
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force_refresh = force_refresh
        self.data_dir = Path(create_synthetic_cache_dir(data_dir, gan_path, rate))

    def prepare_data(self):
        if self.force_refresh or not self.data_dir.exists() or not any(self.data_dir.iterdir()):
            print(f"🛠️ Generating synthetic images at {self.data_dir} ...")
            self.generate_fake_images()
        else:
            print(f"✅ Using cached synthetic images from {self.data_dir}")

    @torch.no_grad()
    def generate_fake_images(self):
        z_dim, c_dim, device = self.G.z_dim, self.G.c_dim, next(self.G.parameters()).device
        self.data_dir.mkdir(parents=True, exist_ok=True)
        num_per_class = self.total_num // self.num_classes
        for label in range(self.num_classes):
            class_dir = self.data_dir / f'class_{label}'
            class_dir.mkdir(parents=True, exist_ok=True)
            for start in tqdm(range(0, num_per_class, self.batch_size), desc=f'Class {label}', leave=False):
                current_batch_size = min(self.batch_size, num_per_class - start)
                z = torch.randn(current_batch_size, z_dim, device=device)
                c = torch.full((current_batch_size, c_dim), float(label), device=device)
                fake_images = self.G(z, c, truncation_psi=0.7, noise_mode='const')
                fake_images = (fake_images.clamp(-1, 1) + 1) / 2
                for i in range(current_batch_size):
                    filename = f'{start + i:05d}_{label}.png'
                    filepath = class_dir / filename
                    save_image(fake_images[i], str(filepath))

    def setup(self, stage='fit'):
        fake_dataset = _SyntheticDataset(self.data_dir)
        if self.rate > 0:
            real_dataset_train = FER2013CSV(self.real_path, split='Training')
            real_size = int(self.rate * self.total_num)
            fake_size = self.total_num - real_size
            fake_subset, _ = random_split(fake_dataset, [fake_size, len(fake_dataset) - fake_size])
            real_subset, _ = random_split(real_dataset_train, [real_size, len(real_dataset_train) - real_size])
            self.train_dataset = ConcatDataset([real_subset, fake_subset])
        else:
            self.train_dataset = fake_dataset

        self.val_dataset = FER2013CSV(self.real_path, split='PublicTest')
        self.test_dataset = FER2013CSV(self.real_path, split='PrivateTest')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

import hashlib
import math
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset
from torchmetrics.classification import Accuracy, F1Score, AUROC, Precision, Recall
from torchmetrics.classification import ConfusionMatrix
from torchvision import transforms, models
from torchvision.utils import save_image
from tqdm import tqdm

import legacy

warnings.filterwarnings('ignore', category=FutureWarning, module='traitlets')
warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics')


def create_synthetic_cache_dir(base_dir: str, gan_path: str) -> str:
    """
    Generate a unique cache directory name for synthetic images based on GAN and real data ratio.

    Args:
        base_dir (str): Root directory where cached images will be stored.
        gan_path (str): Path to the StyleGAN .pkl model.

    Returns:
        str: Unique cache directory path under `base_dir`.
    """
    gan_hash = hashlib.md5(gan_path.encode()).hexdigest()[:8]
    return str(Path(base_dir) / f"gan_{gan_hash}")


class LitClassification(L.LightningModule):
    def __init__(self, num_classes: int = 7, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.set_model()

        self.train_loss_ema = None
        self.train_acc_ema = None
        self.ema_alpha = 0.1

        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.auroc = AUROC(task='multiclass', num_classes=num_classes, average='macro')
        self.precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.confmat = ConfusionMatrix(task='multiclass', num_classes=num_classes)

    def set_model(self) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits, y)

        if self.train_loss_ema is None:
            self.train_loss_ema = loss.detach()
        else:
            self.train_loss_ema = self.ema_alpha * loss.detach() + (1 - self.ema_alpha) * self.train_loss_ema

        if self.train_acc_ema is None:
            self.train_acc_ema = acc.detach()
        else:
            self.train_acc_ema = self.ema_alpha * acc.detach() + (1 - self.ema_alpha) * self.train_acc_ema

        self.log('train_loss', self.train_loss_ema, prog_bar=True)
        self.log('train_acc', self.train_acc_ema, prog_bar=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', self.accuracy(logits, y), prog_bar=True)
        self.log('val_f1', self.f1(logits, y), prog_bar=True)
        self.log('val_auroc', self.auroc(logits, y), prog_bar=False)
        self.log('val_precision', self.precision(logits, y), prog_bar=False)
        self.log('epoch', self.current_epoch, prog_bar=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        probas = torch.softmax(logits, dim=1)

        self.log('test_acc', self.accuracy(preds, y), prog_bar=True)
        self.log('test_f1', self.f1(preds, y), prog_bar=True)
        self.log('test_auroc', self.auroc(probas, y), prog_bar=False)
        self.log('test_precision', self.precision(preds, y), prog_bar=False)
        self.log('test_recall', self.recall(preds, y), prog_bar=False)
        self.log('test_loss', F.cross_entropy(logits, y), prog_bar=False)

        self.confmat.update(preds, y)

    def on_test_epoch_end(self):
        cm = self.confmat.compute()
        print('\n🧩 Confusion Matrix:')
        print(cm)

    def lr_schedule_fn(self, current_epoch: int) -> float:
        warmup_epochs = 5
        total_epochs = self.trainer.max_epochs
        if current_epoch < warmup_epochs:
            return (current_epoch + 1) / warmup_epochs
        else:
            progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(progress * math.pi))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=self.lr_schedule_fn)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': 'val_loss',
            },
        }


class ResNet50(LitClassification):
    def set_model(self):
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, self.hparams.num_classes)
        return self.model


class ResNet18(LitClassification):
    def set_model(self):
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, self.hparams.num_classes)
        return self.model


class EfficientNetB0(LitClassification):
    def set_model(self):
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.hparams.num_classes)
        return self.model


class MobileNetV3(LitClassification):
    def set_model(self):
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, self.hparams.num_classes)
        return self.model


class ShuffleNetV2(LitClassification):
    def set_model(self):
        self.model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, self.hparams.num_classes)
        return self.model


class DenseNet121(LitClassification):
    def set_model(self):
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Linear(self.model.classifier.in_features, self.hparams.num_classes)
        return self.model


class ConvNeXtTiny(LitClassification):
    def set_model(self):
        self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, self.hparams.num_classes)
        return self.model


class _SyntheticDataset(Dataset):
    def __init__(self, image_dir: str):
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), int(img_path.name.split('_')[-1].split('.')[0])


class FER2013CSV(Dataset):
    def __init__(self, csv_file, split='Training'):
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


class FERData(L.LightningDataModule):
    def __init__(
        self,
        generator,
        data_dir,
        gan_path='tmp_fake',
        real_path=None,
        rate=0.0,
        total_num=28708,
        num_classes=7,
        batch_size=64,
        num_workers=4,
        prefetch_factor=2,
    ):
        """
        Lightning DataModule for training with synthetic and optionally real data.

        Args:
            generator: A pre-trained StyleGAN generator (e.g., from stylegan2-ada).
            data_dir (str): Directory to save or load generated synthetic images.
            gan_path (str): Path to store images generated by the GAN.
            real_path (str): Path to the FER-2013 CSV file (required if `rate` > 0).
            rate (float): Proportion (0.0 to 1.0) of real data to include in the training dataset.
                          If 0.0, only synthetic data will be used. For example, rate=0.2 means
                          20% of training data will be real, and 80% will be synthetic.
            total_num (int): Total number of training samples train.
                             This determines how many synthetic + real samples will be used in training.
                             It does not include the test set.
            num_classes (int): Number of classification labels (default is 7 for FER-2013).
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of subprocesses to use for data loading.
        """
        super().__init__()
        assert 0.0 <= rate <= 1.0
        if rate > 0.0:
            assert real_path is not None

        self.G = generator
        self.data_dir = data_dir
        self.real_path = real_path
        self.gan_path = gan_path
        self.rate = rate
        self.total_num = total_num
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def prepare_data(self):

        self.data_dir = create_synthetic_cache_dir('tmp_fake', self.gan_path)
        data_path = Path(self.data_dir)
        if not data_path.exists() or not any(data_path.iterdir()):
            print(f"🛠️ Generating synthetic images at {data_path} ...")
            self.generate_fake_images()
        else:
            print(f"✅ Using cached synthetic images from {data_path}")

    @torch.no_grad()
    def generate_fake_images(self):
        """
        Generate synthetic images using the provided StyleGAN generator.

        This method will generate `total_num` images in total, evenly distributed across `num_classes` labels.
        Images are saved into subdirectories like `class_0/`, `class_1/`, etc., under `data_dir`.

        The filename format is: {index}_{class_label}.png, e.g., "00001_2.png"

        Note:
            - Images are generated with truncation_psi=0.7 and noise_mode='const'.
            - The generator is assumed to accept label-conditioned input `c`.
        """
        z_dim, c_dim, device = self.G.z_dim, self.G.c_dim, next(self.G.parameters()).device
        print(f"running on {device}")
        data_path = Path(self.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        num_per_class = self.total_num // self.num_classes

        for label in range(self.num_classes):
            class_dir = data_path / f'class_{label}'
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

    def setup(self, stage: str = 'fit'):
        fake_dataset = _SyntheticDataset(self.data_dir)

        if self.rate > 0:
            real_size = int(self.rate * self.total_num)
            fake_size = self.total_num - real_size

            real_dataset_train = FER2013CSV(self.real_path, split='Training')
            fake_subset, _ = random_split(fake_dataset, [fake_size, len(fake_dataset) - fake_size])
            real_subset, _ = random_split(real_dataset_train, [real_size, len(real_dataset_train) - real_size])

            self.train_dataset = ConcatDataset([real_subset, fake_subset])
        else:
            self.train_dataset = fake_dataset

        self.test_dataset = FER2013CSV(self.real_path, split='PrivateTest')
        self.val_dataset = FER2013CSV(self.real_path, split='PublicTest')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            pin_memory=True,
        )


def run_experiments(
    gans,
    real_rates: list[float],
    model_class: list[type[LitClassification]],
    real_csv_path,
    save_dir: str | None = None,
):

    for gan_path in gans:
        all_results = []
        with open(gan_path, 'rb') as f:
            gan = legacy.load_network_pkl(f)['G_ema'].to('cuda')

        for rate in real_rates:
            for class_ in model_class:
                model = class_(num_classes=7)
                label = f"{class_.__name__}_gan{Path(gan_path).stem}"
                print(f"\n🚀 Training: {label} at rate {rate:.2f}...")

                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    mode='min',
                )
                checkpoint = ModelCheckpoint(
                    monitor='val_f1',
                    mode='max',
                    save_top_k=1,
                    dirpath=Path(save_dir or './') / 'checkpoints' / Path(gan_path).stem / str(rate) / class_.__name__,
                    filename='{epoch}-{val_f1:.3f}',
                )
                logger = TensorBoardLogger(
                    save_dir=Path(save_dir or './') / 'lightning_logs' / Path(gan_path).stem,
                    name=str(rate),
                    version=class_.__name__,
                )
                lr_monitor = LearningRateMonitor(logging_interval='epoch')

                data_module = FERData(
                    gan,
                    data_dir='tmp_fake',
                    real_path=real_csv_path,
                    rate=rate,
                    batch_size=128,
                    prefetch_factor=4,
                    num_workers=8,
                )

                trainer = Trainer(
                    max_epochs=100,
                    accelerator='auto',
                    precision='16-mixed',
                    devices='auto',
                    callbacks=[early_stop, checkpoint, lr_monitor],
                    log_every_n_steps=10,
                    logger=logger,
                )

                trainer.fit(model, datamodule=data_module)

                best_model = class_.load_from_checkpoint(checkpoint.best_model_path)
                test_result = trainer.test(best_model, datamodule=data_module, verbose=False)[0]
                final_epoch = trainer.current_epoch + 1

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                all_results.append(
                    {
                        'timestamp': timestamp,
                        'model': class_.__name__,
                        'gan': Path(gan_path).stem,
                        'rate': rate,
                        'epochs': final_epoch,
                        'acc': test_result['test_acc'],
                        'f1': test_result['test_f1'],
                        'auroc': test_result['test_auroc'],
                        'precision': test_result['test_precision'],
                        'recall': test_result['test_recall'],
                        'loss': test_result['test_loss'],
                    }
                )

                print(f"✅ Finished {label}")

        csv_path = Path(save_dir or './') / f'{Path(gan_path).stem}_results.csv'
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False)
        print(f"\n📄 All results saved to {csv_path}")

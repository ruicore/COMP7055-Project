import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FER2013

from config import H


class FER2013Dataset:
    batch_size: int = 64
    train_rate: float = 0.8
    transform: transforms.Compose = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )

    def __init__(self):
        full_dataset = FER2013(root=H.fer2013_path, split='train', transform=self.transform)

        train_size = int(self.train_rate * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(
            FER2013(root=H.fer2013_path, split='test', transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False,
        )

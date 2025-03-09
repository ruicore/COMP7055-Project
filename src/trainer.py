import logging
import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import H
from models.resnet import ResNet50Model
from utils.dataset import FER2013Dataset

logging.basicConfig(level=logging.DEBUG)


@dataclass
class ModelTrainer:

    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
        self.model = ResNet50Model().to(H.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=H.learning_rate)
        self.dataset = FER2013Dataset()

    def train(self) -> None:
        no_improve_epochs, best_val_loss = 0, float('inf')
        for epoch in range(H.max_epochs):
            start = time.time()
            logging.info(f'Starting epoch {epoch + 1} at {time.strftime("%H:%M:%S")}')
            self.model.train()
            train_loss = 0
            for images, labels in self.dataset.train_loader:
                images, labels = images.to(H.device), labels.to(H.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in self.dataset.val_loader:
                    images, labels = images.to(H.device), labels.to(H.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            logging.info(
                f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {time.time() - start:.2f}s'
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                torch.save(self.model.state_dict(), f'./{H.fer2013_path} best_model.pth')
                logging.info(f'Best model saved, val loss: {val_loss:.4f}')
            else:
                no_improve_epochs += 1
                logging.info(f'Validation loss did not improve ({no_improve_epochs}/{H.early_stopping_patience})')

            if no_improve_epochs >= H.early_stopping_patience:
                logging.info('Early stopping triggered, training terminated')
                break

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        all_predictions, all_labels = [], []
        all_probs = []
        with torch.no_grad():
            for images, labels in self.dataset.test_loader:
                images, labels = images.to(H.device), labels.to(H.device)
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'f1_score': f1_score(all_labels, all_predictions, average='macro'),
            'auc': roc_auc_score(all_labels, np.array(all_probs), multi_class='ovr'),
        }

        return metrics


if __name__ == '__main__':
    real_trainer = ModelTrainer()
    real_trainer.train()
    # real_metrics = real_trainer.evaluate(test_loader)
    # print('Real Data Metrics:', real_metrics)
    #
    # 训练合成数据模型（假设合成数据仍为文件夹格式）
    # synth_trainer = ModelTrainer('resnet50')
    # synth_train_loader = load_data(csv_file='synthetic_data.csv', usage='Training')  # 如果有合成 CSV
    # 或使用文件夹格式
    # from torchvision import datasets
    #
    # synth_train_loader = DataLoader(
    #     datasets.ImageFolder(
    #         'synthetic_data',
    #         transform=transforms.Compose([transforms.Grayscale(), transforms.Resize((48, 48)), transforms.ToTensor()]),
    #     ),
    #     batch_size=64,
    #     shuffle=True,
    # )
    # synth_trainer.train(synth_train_loader, epochs=10)
    # synth_metrics = synth_trainer.evaluate(test_loader)
    # print('Synthetic Data Metrics:', synth_metrics)

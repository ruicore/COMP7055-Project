import hashlib
import logging
import re
from pathlib import Path

import legacy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchmetrics.classification import AUROC, Accuracy, ConfusionMatrix, F1Score, Precision, Recall
from torchvision import models, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from data import FER2013CSV, FERData

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def run_experiments(gans, real_rates, model_classes, real_csv_path, output_dir='experiments', max_epochs=30):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for gan_path in gans:
        with open(gan_path, 'rb') as f:
            G = legacy.load_network_pkl(f)['G_ema'].to('cuda')

        for rate in real_rates:
            for ModelClass in model_classes:
                label = f"{ModelClass.__name__}_rate{rate}_gan{Path(gan_path).stem}"
                log_path = output_dir / label
                log_path.mkdir(parents=True, exist_ok=True)

                checkpoint = ModelCheckpoint(
                    monitor='val_f1', mode='max', save_top_k=1, dirpath=log_path, filename='{epoch}-{val_f1:.3f}'
                )

                early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

                model = ModelClass(num_classes=7)
                logging.info(f"🚀 Training: {label}")

                data_module = FERData(
                    generator=G,
                    gan_path=gan_path,
                    data_dir='tmp_fake',
                    real_path=real_csv_path,
                    rate=rate,
                    force_refresh=False,
                )

                trainer = Trainer(
                    max_epochs=max_epochs,
                    accelerator='auto',
                    devices='auto',
                    callbacks=[early_stop, checkpoint],
                    log_every_n_steps=10,
                )

                trainer.fit(model, datamodule=data_module)

                best_model = ModelClass.load_from_checkpoint(checkpoint.best_model_path)
                test_result = trainer.test(best_model, datamodule=data_module, verbose=False)[0]

                summary = {
                    'label': label,
                    'best_ckpt': str(checkpoint.best_model_path),
                    'test_acc': float(test_result['test_acc']),
                    'test_f1': float(test_result['test_f1']),
                    'test_loss': float(test_result['test_loss']),
                }

                results.append(summary)

                with open(log_path / 'result.yaml', 'w') as f:
                    yaml.dump(summary, f)

    with open(output_dir / 'summary.yaml', 'w') as f:
        yaml.dump(results, f)


def plot_results(metrics_file='experiment_results.txt'):
    pattern = re.compile(
        r'(?P<label>.*?) - Acc: (?P<acc>[\d.]+), F1: (?P<f1>[\d.]+), AUROC: (?P<auroc>[\d.]+), '
        r'Precision: (?P<precision>[\d.]+), Recall: (?P<recall>[\d.]+), Loss: (?P<loss>[\d.]+)'
    )

    records = []
    with open(metrics_file) as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                records.append(match.groupdict())

    df = pd.DataFrame(records)
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)

    metrics = ['acc', 'f1', 'auroc', 'precision', 'recall']
    for metric in metrics:
        plt.figure(figsize=(12, 5))
        plt.title(f'Comparison: {metric.upper()}')
        plt.xticks(rotation=45, ha='right')
        plt.bar(df['label'], df[metric])
        plt.tight_layout()
        plt.savefig(f'{metric}_comparison.png')
        plt.close()

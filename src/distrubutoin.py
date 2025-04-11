import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap.umap_ as umap
from pytorch_lightning import seed_everything
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from single import FER2013CSV, ConvNeXtTiny, SyntheticDataset


class FeatureExtractor(ConvNeXtTiny):
    def __init__(self, ckpt_path):
        super().__init__()
        self.load_state_dict(torch.load(ckpt_path)['state_dict'])
        self.eval()

    def forward(self, x):
        x = self.model.features(x)
        x = x.mean(dim=[2, 3])
        return x


@torch.no_grad()
def extract_features(dataloader, model):
    all_feats, all_labels = [], []
    model.eval()

    for imgs, labels in tqdm(dataloader, desc='Extracting features'):
        feats = model(imgs.cuda())
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    features = np.concatenate(all_feats)
    labels = np.concatenate(all_labels)
    return features, labels


def plot_embedding(
    real_features,
    gan_features,
    real_labels,
    gan_labels,
    reducer,
    name,
    model_label,
    class_names=None,
    save_dir='outputs',
):
    combined = np.concatenate([real_features, gan_features], axis=0)
    labels = np.concatenate([real_labels, gan_labels], axis=0)
    reduced = reducer.fit_transform(combined)
    num_classes = len(np.unique(labels))

    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_classes):
        real_idx = np.where(real_labels == i)[0]
        gan_idx = np.where(gan_labels == i)[0]
        label = class_names[i] if class_names else f'Class {i}'

        plt.figure(figsize=(8, 6))
        plt.scatter(
            reduced[real_idx, 0], reduced[real_idx, 1], label=f'{label} (Real)', alpha=0.6, marker='o', c='blue'
        )
        plt.scatter(
            reduced[len(real_labels) + gan_idx, 0],
            reduced[len(real_labels) + gan_idx, 1],
            label=f'{label} ({model_label})',
            alpha=0.6,
            marker='x',
            c='red',
        )

        plt.legend()
        plt.title(f'{name}: {label} - {model_label}')
        plt.grid(True)

        filename = f'{name.lower()}_{label.replace(" ", "_")}_{model_label.replace(" ", "_")}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()


def compare_all_embeddings(class_names=None, save_dir='outputs'):
    real_model = FeatureExtractor('epoch=44-val_f1=0.509.ckpt').cuda()
    gan_models = [
        (FeatureExtractor('epoch=5-val_f1=0.243.ckpt').cuda(), 'GAN 0.0'),
        (FeatureExtractor('epoch=14-val_f1=0.328.ckpt').cuda(), 'GAN 0.1'),
        (FeatureExtractor('epoch=15-val_f1=0.404.ckpt').cuda(), 'GAN 0.3'),
        (FeatureExtractor('epoch=35-val_f1=0.443.ckpt').cuda(), 'GAN 0.5'),
    ]

    fake_loader = DataLoader(SyntheticDataset('gan_2859e4b2_rate0'), batch_size=64, num_workers=8)
    real_loader = DataLoader(FER2013CSV('fer2013.csv'), batch_size=64, num_workers=8)

    real_features, real_labels = extract_features(real_loader, real_model)

    for gan_model, label in gan_models:
        gan_features, gan_labels = extract_features(fake_loader, gan_model)

        plot_embedding(
            real_features,
            gan_features,
            real_labels,
            gan_labels,
            TSNE(n_components=2, random_state=42),
            name='t-SNE',
            model_label=label,
            class_names=class_names,
            save_dir=os.path.join(save_dir, 'tsne'),
        )

        plot_embedding(
            real_features,
            gan_features,
            real_labels,
            gan_labels,
            umap.UMAP(n_components=2, random_state=42),
            name='UMAP',
            model_label=label,
            class_names=class_names,
            save_dir=os.path.join(save_dir, 'umap'),
        )

        plot_embedding(
            real_features,
            gan_features,
            real_labels,
            gan_labels,
            PCA(n_components=2),
            name='PCA',
            model_label=label,
            class_names=class_names,
            save_dir=os.path.join(save_dir, 'pca'),
        )


if __name__ == '__main__':
    seed_everything(42)

    compare_all_embeddings(
        [
            'Angry',
            'Disgust',
            'Fear',
            'Happy',
            'Sad',
            'Surprise',
            'Neutral',
        ],
        save_dir='outputs',
    )
    print(type(PCA(n_components=2)).__name__)

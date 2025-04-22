import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap.umap_ as umap
from PIL import Image
from pytorch_lightning import seed_everything
from scipy import linalg
from scipy.spatial.distance import jensenshannon
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


def custom_fid(real_feats, fake_feats):
    mu1, sigma1 = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def evaluate_fid(
    real_feats,
    real_labels,
    gan_feats,
    gan_labels,
    class_names=None,
    data_label='GAN',
    save_dir: str | Path = 'outputs',
):
    results = []
    classes = np.unique(real_labels)

    for cls in classes:
        real_idx = real_labels == cls
        gan_idx = gan_labels == cls

        real_subset = real_feats[real_idx]
        gan_subset = gan_feats[gan_idx]

        if len(real_subset) < 2 or len(gan_subset) < 2:
            fid = float('nan')
        else:
            fid = custom_fid(real_subset, gan_subset)

        label = class_names[cls] if class_names else f"Class {cls}"
        results.append((label, fid))

    print('\nPer-class FID results:')
    print('Class\t\tFID')
    print('--------------------------')
    for label, fid in results:
        print(f"{label:<10}\t{fid:.2f}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"fid_per_class_{data_label}.csv"
    df = pd.DataFrame(results, columns=['Class', 'FID'])
    df.to_csv(csv_path, index=False)

    labels, fids = zip(*results)
    plt.figure(figsize=(10, 5))
    plt.bar(labels, fids, color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('FID')
    plt.title(f'Per-class FID - {data_label}')
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"fid_per_class_bar_{data_label}.png", dpi=300)
    plt.close()

    return results


def compute_js_on_tsne(real_tsne, gan_tsne, bins=50):
    all_points = np.vstack([real_tsne, gan_tsne])
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)

    H_real, _, _ = np.histogram2d(real_tsne[:, 0], real_tsne[:, 1], bins=bins, range=[[x_min, x_max], [y_min, y_max]])
    H_gan, _, _ = np.histogram2d(gan_tsne[:, 0], gan_tsne[:, 1], bins=bins, range=[[x_min, x_max], [y_min, y_max]])

    P = H_real.flatten() / H_real.sum()
    Q = H_gan.flatten() / H_gan.sum()

    return jensenshannon(P, Q)


def plot_embedding(
    real_features,
    gan_features,
    real_labels,
    gan_labels,
    reducer,
    class_names=None,
    save_dir='outputs',
    data_label='GAN',
    method_name='method',
    metrics_log=None,
):
    combined = np.concatenate([real_features, gan_features], axis=0)
    labels = np.concatenate([real_labels, gan_labels], axis=0)
    reduced = reducer.fit_transform(combined)
    num_classes = len(np.unique(labels))

    js_dist = None
    if method_name == 'TSNE':
        js_dist = compute_js_on_tsne(reduced[: len(real_labels)], reduced[len(real_labels) :])
        print(f"\n{type(reducer).__name__} Jensen-Shannon Distance: {js_dist:.4f} on {data_label}")

    if metrics_log is not None:
        metrics_log.append({'Data': data_label, 'Method': method_name, 'JS_Distance': js_dist})

    Path(save_dir).mkdir(parents=True, exist_ok=True)

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
            label=f'{label}',
            alpha=0.6,
            marker='x',
            c='red',
        )

        plt.legend()
        plt.title(f'{method_name}: {label}')
        plt.grid(True)

        plt.savefig(Path(save_dir) / f'{label}.png', dpi=300)
        plt.close()


def stitch_class_images(image_dir, output_path, grid_size=(2, 4)):
    files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    images = [Image.open(os.path.join(image_dir, f)) for f in files]

    if not images:
        return

    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    grid_cols, grid_rows = grid_size

    stitched_img = Image.new('RGB', (max_width * grid_cols, max_height * grid_rows), color='white')

    for idx, img in enumerate(images):
        x = (idx % grid_cols) * max_width
        y = (idx // grid_cols) * max_height
        stitched_img.paste(img, (x, y))

    stitched_img.save(output_path)


def compare_all_embeddings(class_names, save_dir='outputs'):
    real_model = FeatureExtractor('epoch=44-val_f1=0.509.ckpt').cuda()
    fake_loaders = [
        (DataLoader(SyntheticDataset('tmp_fake/gan_2859e4b2_rate0'), batch_size=64, num_workers=8), 'GAN0.0'),
        (DataLoader(SyntheticDataset('tmp_fake/gan_2859e4b2_rate10'), batch_size=64, num_workers=8), 'GAN0.1'),
        (DataLoader(SyntheticDataset('tmp_fake/gan_2859e4b2_rate30'), batch_size=64, num_workers=8), 'GAN0.3'),
        (DataLoader(SyntheticDataset('tmp_fake/gan_2859e4b2_rate50'), batch_size=64, num_workers=8), 'GAN0.5'),
    ]

    real_features, real_labels = extract_features(
        DataLoader(
            FER2013CSV('fer2013.csv'),
            batch_size=64,
            num_workers=8,
        ),
        real_model,
    )
    metrics_log = []

    for fake_loader, data_label in fake_loaders:
        gan_features, gan_labels = extract_features(fake_loader, real_model)
        evaluate_fid(
            real_features,
            real_labels,
            gan_features,
            gan_labels,
            class_names=class_names,
            data_label=data_label,
            save_dir=Path(save_dir) / data_label / 'fid',
        )

        for reducer in [
            TSNE(n_components=2, random_state=42),
            PCA(n_components=2),
            umap.UMAP(n_components=2, random_state=42),
        ]:
            method = type(reducer).__name__
            subdir = Path(save_dir) / data_label / method.lower()
            plot_embedding(
                real_features,
                gan_features,
                real_labels,
                gan_labels,
                reducer,
                class_names=class_names,
                save_dir=subdir,
                data_label=data_label,
                method_name=method,
                metrics_log=metrics_log,
            )
            stitch_class_images(subdir, Path(save_dir) / f'{method.lower()}_summary.png')

    df_metrics = pd.DataFrame(metrics_log)
    df_metrics.to_csv(Path(save_dir) / 'embedding_metrics.csv', index=False)

    pivot = df_metrics.pivot(index='Data', columns='Method', values='JS_Distance')
    plt.figure(figsize=(8, 5))
    plt.title('Jensen-Shannon Distance Heatmap')
    sns.heatmap(pivot, annot=True, cmap='coolwarm', fmt='.4f')
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'js_distance_heatmap.png', dpi=300)
    plt.close()


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

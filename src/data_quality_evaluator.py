import numpy as np
import torch
import torch.nn.functional as F
from inception_v3 import InceptionV3
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision import transforms


class DataQualityEvaluator:
    """合成数据质量评估器"""

    def __init__(self, real_csv_file: str, synth_data_dir: str, batch_size: int = 64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
            ]
        )
        self.real_loader = self._load_real_data(real_csv_file)
        self.synth_loader = self._load_synth_data(synth_data_dir)
        self.inception = InceptionV3().to(self.device).eval()

    def _load_real_data(self, csv_file: str) -> DataLoader:
        dataset = FER2013Dataset(csv_file, transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def _load_synth_data(self, data_dir: str) -> DataLoader:
        # 假设合成数据仍以文件夹形式存储
        dataset = datasets.ImageFolder(data_dir, transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def compute_fid(self) -> float:
        """计算 Fréchet Inception Distance"""
        real_features = self._extract_features(self.real_loader)
        synth_features = self._extract_features(self.synth_loader)

        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = np.mean(synth_features, axis=0), np.cov(synth_features, rowvar=False)

        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def _extract_features(self, loader: DataLoader) -> np.ndarray:
        features = []
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(self.device)
                imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear')
                feats = self.inception(imgs).cpu().numpy()
                features.append(feats)
        return np.concatenate(features, axis=0)

    def compute_ssim(self) -> float:
        """计算平均 SSIM"""
        real_imgs = [img for img, _ in self.real_loader.dataset[:100]]
        synth_imgs = [img for img, _ in self.synth_loader.dataset[:100]]
        ssim_scores = [ssim(real_imgs[i].numpy()[0], synth_imgs[i].numpy()[0]) for i in range(100)]
        return np.mean(ssim_scores)


# 使用示例
if __name__ == '__main__':
    evaluator = DataQualityEvaluator(
        real_csv_file='fer2013/fer2013.csv', synth_data_dir='synthetic_data'  # 假设 CSV 文件路径
    )
    fid_score = evaluator.compute_fid()
    ssim_score = evaluator.compute_ssim()
    print(f"FID: {fid_score:.2f}, SSIM: {ssim_score:.2f}")

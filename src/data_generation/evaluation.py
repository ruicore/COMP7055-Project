from pytorch_fid import fid_score
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim


class SyntheticDataEvaluator:
    def __init__(self, real_dir='FER2013/train', fake_dir='generated_faces'):
        self.real_dir = real_dir
        self.fake_dir = fake_dir

    def compute_ssim(self, img1, img2):
        return ssim(img1, img2)

    def compute_fid(self):
        fid = fid_score.calculate_fid_given_paths([self.real_dir, self.fake_dir], 50, 'cuda', 2048)
        print(f"FID Score: {fid}")
        return fid

    def compute_kl_divergence(self, real_dist, fake_dist):
        kl_div = entropy(fake_dist, real_dist)
        print(f"KL Divergence: {kl_div}")
        return kl_div


if __name__ == '__main__':
    evaluator = SyntheticDataEvaluator()
    evaluator.compute_fid()

# echo "torch
# torchvision
# torchaudio
# timm
# numpy
# pandas
# matplotlib
# seaborn
# opencv-python
# pillow
# tqdm
# scikit-learn
# scipy
# pytorch-fid
# dnnlib
# legacy
# click
# requests
# pyspng
# ninja" > requirements.txt
# pip install -r requirements.txt

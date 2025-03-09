import os

import dnnlib
import legacy
import PIL.Image
import torch


class StyleGANGenerator:
    def __init__(self, model_path='pretrained/stylegan2-ada.pkl', output_dir='generated_faces', num_images=1000):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.output_dir = output_dir
        self.num_images = num_images
        os.makedirs(self.output_dir, exist_ok=True)

        self._load_model()

    def _load_model(self):
        """加载 StyleGAN2-ADA 预训练模型"""
        print('Loading StyleGAN2 model...')
        with dnnlib.util.open_url(self.model_path) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)  # 加载生成器

    def generate_images(self):
        """生成合成面部表情数据"""
        print(f"Generating {self.num_images} synthetic images...")
        for i in range(self.num_images):
            z = torch.randn(1, self.G.z_dim).to(self.device)
            img = self.G(z, None)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = img.permute(0, 2, 3, 1).cpu().numpy()[0]

            img_pil = PIL.Image.fromarray(img)
            img_pil.save(f"{self.output_dir}/synthetic_{i}.png")


if __name__ == '__main__':
    generator = StyleGANGenerator()
    generator.generate_images()

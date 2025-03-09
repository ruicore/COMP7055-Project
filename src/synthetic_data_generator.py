# synthetic_data_generator.py
import os
from typing import List, Tuple

import torch
import torchvision.transforms as transforms
from stylegan2_pytorch import Model  # 假设使用 StyleGAN2 的 PyTorch 实现


class SyntheticDataGenerator:
    """合成数据生成器接口，使用 StyleGAN 生成表情图像"""

    def __init__(self, model_path: str, output_dir: str, num_samples: int = 30000):
        self.model_path = model_path  # 预训练 StyleGAN 模型路径
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()

    def _load_model(self) -> Model:
        """加载预训练 StyleGAN 模型"""
        model = Model().to(self.device)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

    def generate_images(self, labels: List[int]) -> List[Tuple[str, int]]:
        """生成带有表情标签的合成图像"""
        os.makedirs(self.output_dir, exist_ok=True)
        image_paths = []

        # 假设每个表情类别生成均衡数量
        samples_per_class = self.num_samples // 7
        with torch.no_grad():
            for label in labels:  # 0-6 表示 7 种表情
                for i in range(samples_per_class):
                    noise = torch.randn(1, 512).to(self.device)  # StyleGAN 输入噪声
                    # 条件输入：将标签嵌入到生成过程中（需根据具体实现调整）
                    fake_img = self.model(noise, label=label)
                    img_path = os.path.join(self.output_dir, f"synth_{label}_{i}.png")
                    transforms.ToPILImage()(fake_img.squeeze()).save(img_path)
                    image_paths.append((img_path, label))
        return image_paths


# 使用示例
if __name__ == '__main__':
    generator = SyntheticDataGenerator(
        model_path='path/to/pretrained_stylegan.pth', output_dir='synthetic_data', num_samples=30000
    )
    labels = list(range(7))  # FER-2013 的 7 类表情
    synth_data = generator.generate_images(labels)
    print(f"Generated {len(synth_data)} synthetic images.")

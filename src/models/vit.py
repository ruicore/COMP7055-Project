import timm  # 使用 timm 提供的 ViT 预训练模型
import torch.nn as nn


class ViTModel(nn.Module):
    def __init__(self, num_classes=7):
        super(ViTModel, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def get_vit_model():
    return ViTModel()

import torch.nn as nn
import torchvision.models as models

from config import H


class ResNet50Model(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, H.num_classes)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Linear(self.model.fc.in_features, H.num_classes)

    def forward(self, x):
        return self.model(x)


def get_resnet50_model():
    return ResNet50Model()

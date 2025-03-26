import torch.nn as nn
import torchvision.models as models

from base import LitClassification


class ResNet50(LitClassification):
    def set_model(self):
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, self.hparams.num_classes)
        return self.model


class ResNet18(LitClassification):
    def set_model(self):
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, self.hparams.num_classes)
        return self.model


class EfficientNetB0(LitClassification):
    def set_model(self):
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.hparams.num_classes)
        return self.model

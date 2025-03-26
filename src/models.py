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


class MobileNetV3(LitClassification):
    def set_model(self):
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, self.hparams.num_classes)
        return self.model


class ShuffleNetV2(LitClassification):
    def set_model(self):
        self.model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, self.hparams.num_classes)
        return self.model


class DenseNet121(LitClassification):
    def set_model(self):
        self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Linear(self.model.classifier.in_features, self.hparams.num_classes)
        return self.model


class ConvNeXtTiny(LitClassification):
    def set_model(self):
        self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, self.hparams.num_classes)
        return self.model

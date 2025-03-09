from dataclasses import dataclass


@dataclass
class HyperParameters:
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.0001
    device: str = 'mps'

    fer2013_path: str = 'data'
    synthetic_path: str = 'generated_faces'
    num_classes: int = 7

    resnet_pretrained: bool = True
    vit_pretrained: bool = True

    early_stopping_patience: int = 5
    max_epochs: int = 50


H = HyperParameters()

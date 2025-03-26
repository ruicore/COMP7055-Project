import argparse

import yaml

from models import ConvNeXtTiny, DenseNet121, EfficientNetB0, MobileNetV3, ResNet18, ResNet50, ShuffleNetV2
from runner import plot_results, run_experiments


def load_gan_paths(config_file='platform.yaml', platform='local'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config.get(platform, [])


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', choices=['kaggle', 'local'], default='local', help='Run mode platform')
    args = parser.parse_args()

    gans = load_gan_paths(config_file='platform.yaml', platform=args.platform)
    real_rates = [
        0.0,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    model_classes = [
        ResNet18,
        ResNet50,
        EfficientNetB0,
        MobileNetV3,
        ShuffleNetV2,
        DenseNet121,
        ConvNeXtTiny,
    ]
    real_csv_path = '.cache/fer2013.csv'

    run_experiments(gans, real_rates, model_classes, real_csv_path)
    plot_results()


if __name__ == '__main__':
    run()

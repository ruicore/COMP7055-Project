import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import yaml
from run import plot_results, run_experiments

from models import ConvNeXtTiny, DenseNet121, EfficientNetB0, MobileNetV3, ResNet18, ResNet50, ShuffleNetV2

MODEL_MAP = {
    'resnet18': ResNet18,
    'resnet50': ResNet50,
    'efficientnetb0': EfficientNetB0,
    'mobilenetv3': MobileNetV3,
    'shufflenetv2': ShuffleNetV2,
    'densenet121': DenseNet121,
    'convnexttiny': ConvNeXtTiny,
}


def schedule_experiments(
    gans,
    real_rates,
    selected_models,
    real_csv_path,
    outdir='experiments',
    max_models_per_run=3,
    dry_run=False,
):
    selected_classes = [MODEL_MAP[name.lower()] for name in selected_models]

    # Split model classes into chunks
    grouped = [
        selected_classes[i : i + max_models_per_run] for i in range(0, len(selected_classes), max_models_per_run)
    ]

    for i, group in enumerate(grouped):
        print(f"🧪 Batch {i + 1}/{len(grouped)}: {[m.__name__ for m in group]}")
        if not dry_run:
            run_experiments(gans, real_rates, group, real_csv_path, output_dir=Path(outdir) / f"batch{i + 1}")

    if not dry_run:
        plot_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='platform.yaml', help='YAML config file')
    parser.add_argument('--platform', choices=['local', 'kaggle'], default='local', help='Platform key in YAML')
    parser.add_argument('--csv', type=str, default='fer2013.csv', help='Path to FER2013 CSV file')
    parser.add_argument('--max_models', type=int, default=3, help='Max models per batch')
    parser.add_argument('--dry', action='store_true', help='Only print schedule, no training')
    parser.add_argument('--models', nargs='+', default=list(MODEL_MAP.keys()), help='Which models to run')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    gans = config.get(args.platform, [])
    real_rates = [0.0, 0.1, 0.3, 0.5, 1.0]

    schedule_experiments(
        gans=gans,
        real_rates=real_rates,
        selected_models=args.models,
        real_csv_path=args.csv,
        max_models_per_run=args.max_models,
        dry_run=args.dry,
    )

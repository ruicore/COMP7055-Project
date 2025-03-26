from models import EfficientNetB0, ResNet18, ResNet50
from train import plot_results, run_experiments

if __name__ == '__main__':
    gans = ['./network-snapshot-004216x64.pkl']
    real_rates = [0.0, 0.1, 0.3, 0.5, 1.0]
    model_classes = [ResNet18, ResNet50, EfficientNetB0]
    real_csv_path = 'fer2013.csv'

    run_experiments(gans, real_rates, model_classes, real_csv_path)
    plot_results()

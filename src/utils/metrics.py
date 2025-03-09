import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class Metrics:
    @staticmethod
    def compute_metrics(y_true, y_pred):
        y_pred_labels = torch.argmax(F.softmax(y_pred, dim=1), dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()

        accuracy = accuracy_score(y_true, y_pred_labels)
        f1 = f1_score(y_true, y_pred_labels, average='weighted')
        auc = roc_auc_score(y_true, F.softmax(y_pred, dim=1).cpu().numpy(), multi_class='ovr')

        return {'accuracy': accuracy, 'f1_score': f1, 'auc': auc}

import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch.nn import Module
from torchmetrics.classification import AUROC, Accuracy, ConfusionMatrix, F1Score, Precision, Recall


class LitClassification(L.LightningModule):
    def __init__(self, num_classes: int = 7, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.set_model()

        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.auroc = AUROC(task='multiclass', num_classes=num_classes, average='macro')
        self.precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.confmat = ConfusionMatrix(task='multiclass', num_classes=num_classes)


    def set_model(self) -> Module:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(logits, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(logits, y), prog_bar=True)
        self.log('val_f1', self.f1(logits, y), prog_bar=True)
        self.log('val_auroc', self.auroc(logits, y), prog_bar=False)
        self.log('val_precision', self.precision(logits, y), prog_bar=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        probas = torch.softmax(logits, dim=1)

        self.log('test_acc', self.accuracy(preds, y), prog_bar=True)
        self.log('test_f1', self.f1(preds, y), prog_bar=True)
        self.log('test_auroc', self.auroc(probas, y), prog_bar=False)
        self.log('test_precision', self.precision(preds, y), prog_bar=False)
        self.log('test_recall', self.recall(preds, y), prog_bar=False)
        self.log('test_loss', F.cross_entropy(logits, y), prog_bar=False)

        self.confmat.update(preds, y)

    def on_test_epoch_end(self):
        cm = self.confmat.compute()
        print('\n🧩 Confusion Matrix:')
        print(cm)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

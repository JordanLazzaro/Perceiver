import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl


class CIFAR10Classifier(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        
        # self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        # self.val_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        # acc = self.train_accuracy(logits, y)
        self.log('train/loss', loss, prog_bar=True)
        # self.log('train/accuracy', acc)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        # acc = self.val_accuracy(logits, y)
        self.log('val/loss', loss, prog_bar=True)
        # self.log('val/accuracy', acc)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=self.learning_rate/10)
        
        return { "optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train/loss" }
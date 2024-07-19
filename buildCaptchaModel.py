import torch
import torchvision.models as models
import torch.nn as nn
from torchmetrics import Accuracy
import lightning.pytorch as pl
from torchinfo import summary

class CaptchaModel(pl.LightningModule):
    def __init__(self, num_classes=4, num_characters=34, input_channels=1, learning_rate=1e-4):
        super(CaptchaModel, self).__init__()
        self.num_classes = num_classes
        self.num_characters = num_characters
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()
        self.acc_fn = Accuracy(task="multiclass", num_classes=self.num_characters)
        self.automatic_optimization = True
        self.save_hyperparameters()

        # self.acc_mean = 0
        # self.loss_mean = 0

        # self.acc_sum = 0
        # self.loss_sum = 0

        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # We use a Resnet50 architecture and adapt it to our needs
        self.resnet50.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet50.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # print(self.num_characters * self.num_classes)
            nn.Linear(in_features=1024, out_features=self.num_characters * self.num_classes, bias=True),
        )

    def forward(self, x):
        return self.resnet50(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)  # Add L2 regularization
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3, min_lr=5e-5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        # print(logits.reshape(logits.shape[0], self.num_characters, self.num_classes).argmax(1))
        # print(y.reshape(y.shape[0], self.num_characters, self.num_classes).argmax(1))
        acc = self.acc_fn(
            logits.reshape(logits.shape[0], self.num_characters, self.num_classes).argmax(1),
            y.reshape(y.shape[0], self.num_characters, self.num_classes).argmax(1),
        )

        # if batch_idx == 0:
        #     self.loss_mean = loss
        #     self.acc_mean = acc

        # self.loss_mean = 0.1 * loss + 0.9 * self.loss_mean
        # self.acc_mean = 0.1 * acc + 0.9 * self.acc_mean

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        acc = self.acc_fn(
            logits.reshape(logits.shape[0], self.num_characters, self.num_classes).argmax(1),
            y.reshape(y.shape[0], self.num_characters, self.num_classes).argmax(1),
        )

        # self.loss_sum += loss
        # self.acc_sum += acc

        # loss_mean = self.loss_sum/(batch_idx+1)
        # acc_mean = self.acc_sum/(batch_idx+1)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)

        self.log("test_loss", loss)
        

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        logits = self(x)
        return logits

    def set_lr(self, lr):
        self.learning_rate = lr
        return self
    
# BATCH_SIZE = 128
# model = CaptchaModel()
# summary(model, input_size=(BATCH_SIZE, 1, 50, 250))
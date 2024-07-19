from lightning.pytorch import Callback
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor,EarlyStopping
from torchinfo import summary
from buildCaptchaModel import CaptchaModel
from torch.utils.data import DataLoader, Dataset
from CaptchaDataset import CaptchaDataset
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from collections import defaultdict
from torchvision import transforms


class MetricsCallback(Callback):
    
    def __init__(self):
        super().__init__()
        self.metrics = []
        
    def on_train_epoch_end(self, trainer, pl_module):
        self.metrics.append(trainer.logged_metrics.copy())

def train_model(train_dir,test_dir,batch_size):
    # train_dir = "./captchas/train/"
    # test_dir = "./captchas/test/"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} as accelerator")

    # BATCH_SIZE = 128configure_optimizers

    train_dataset = CaptchaDataset(train_dir)
    test_dataset = CaptchaDataset(test_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
    # print(train_dataset)
    # print(test_dataset) 

    print(f"Training Data: {len(train_dataset)} files | Testing Data: {len(test_dataset)} files")
    print(f"Training Dataloader: {len(train_loader)} batches of size {batch_size} | Testing Data: {len(test_loader)} batches of size {batch_size}")
    model = CaptchaModel()
    # summary(model, input_size=(batch_size, 1, 60, 160))

    metrics_callback = MetricsCallback()
    torch.cuda.empty_cache()
    checkpoint = ModelCheckpoint(monitor="val_acc", mode="max")
    lr_monitor = LearningRateMonitor("epoch")
    trainer = pl.Trainer(max_epochs=50, precision=16, log_every_n_steps=1, accelerator=device, callbacks=[checkpoint, lr_monitor, metrics_callback], num_sanity_val_steps=0)
    trainer.fit(model, train_loader, test_loader)
    print("Best model path:", checkpoint.best_model_path, "Best model score:", checkpoint.best_model_score)
    ckpt_path = checkpoint.best_model_path
    model = CaptchaModel.load_from_checkpoint(ckpt_path)
    torch.save(model.state_dict(), f'./weights/ResNet_mytrain_weigthts.pth')

    # Get loss, acc etc for visualisation
    metrics_dict = defaultdict(list)
    for metrics in metrics_callback.metrics:
        for key, value in metrics.items():
            metrics_dict[key].append(value.item())
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot accuracy
    axs[0].plot(metrics_dict['train_acc'], label='Training Accuracy')
    axs[0].plot(metrics_dict['val_acc'], label='Validation Accuracy')
    axs[0].set_title('Accuracy over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Plot loss
    axs[1].plot(metrics_dict['train_loss'], label='Training Loss')
    axs[1].plot(metrics_dict['val_loss'], label='Validation Loss')
    axs[1].set_title('Loss over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_dir = "./images/train/"
    test_dir = "./images/test/"
    batch_size = 256
    train_model(train_dir=train_dir,test_dir=test_dir,batch_size=batch_size)



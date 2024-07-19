import torch
import lightning.pytorch as pl
import os
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


NUMBER = ['0','1','2','3','4','5','6','7','8','9']
UPPER = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
CHARSET = NUMBER+UPPER
NUM = [i for i in range(len(CHARSET))]
class CaptchaDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.captcha_len = 4
        self.files = [f for f in os.listdir(self.data_dir) if f.endswith(".png") ]#or f.endswith(".png")
        self.labels = [img.split(".")[0].split("_")[1] for img in self.files]
        # print(self.labels)
        self.encoding_dict = dict(zip(CHARSET,NUM))
    
        self.decoding_dict = dict(zip(NUM,CHARSET))

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.7570), (0.3110))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.files[idx])
        image = self.transform(Image.open(image_path).resize((160, 60)).convert("RGB"))
        if idx<100:
            Image.open(image_path).resize((160, 60)).convert("RGB").show()
            img = transforms.ToPILImage()(image)
            img.show()
        label = self.labels[idx]
        onehot_label = self.to_onehot(label)
        return image, onehot_label

    def to_onehot(self, label):
        onehot = torch.zeros((len(self.encoding_dict), self.captcha_len), dtype=torch.float32)
        # print(label)
        for col, letter in enumerate(label):
            onehot[self.encoding_dict[letter], col] = 1
        return onehot.reshape(-1)

    def get_pred(self, encoded_vector):
        label = []
        encoded_vector = encoded_vector.reshape(len(self.encoding_dict), self.captcha_len).argmax(0)
        for key in encoded_vector.detach().cpu().numpy():
            label.append(self.decoding_dict[key])
        return "".join(label)

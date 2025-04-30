import torch 
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from math import log2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import kagglehub

DATASET = kagglehub.dataset_download("emircanakyuz/womens-clothing-dataset")
START_TRAIN_IMG_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-3
BATCH_SIZE = [256, 256, 128, 64, 32, 16]
CHANNELS_IMG = 3
Z_DIM = 512
W_DIM = 512
IN_CHANNELS = 512
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZE)

def get_loader(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize([0.5] * CHANNELS_IMG, [0.5] * CHANNELS_IMG)
    ])
    batch_size = BATCH_SIZE[int(log2(image_size/4))]                  
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    return dataset, loader

def check_loader():
    _, loader = get_loader(128)
    batch = next(iter(loader))  # Get a batch of data
    cloth, _ = batch   # Unpack the images and labels
    print(f"Shape of cloth tensor: {cloth.shape}")  # Debugging line

    if cloth.dim() != 4:
        raise ValueError(f"Expected a 4D tensor, but got {cloth.dim()}D tensor with shape {cloth.shape}")

    _, ax = plt.subplots(3, 3, figsize=(8, 8))
    plt.suptitle('Some real images')
    ind = 0
    for i in range(3):
        for j in range(3):
            ax[i][j].imshow(cloth[ind].permute(1, 2, 0).cpu().numpy())
            ax[i][j].axis('off')
            ind += 1
    plt.show()

check_loader()  # check loader

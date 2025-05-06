import os
import torch
from torch import optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from model import Generator, Discriminator, TrainConfig, save_checkpoint, save_samples
from tqdm import tqdm
import zipfile
import kagglehub

# Hyperparameters
# Load training configuration
config = TrainConfig()

# Use TrainConfig parameters
latent_dim = config.latent_dim
style_dim = config.style_dim
channels = config.generator_channels  # <- renamed to clarify it's for Generator
image_size = config.image_size
batch_size = config.batch_size
num_epochs = config.num_epochs
lr = config.lr
device = config.device
save_dir = config.save_dir
data_path = config.data_path

os.makedirs(save_dir, exist_ok=True)

# Download dataset using kagglehub
DATASET = kagglehub.dataset_download("emircanakyuz/womens-clothing-dataset")

# Dataset and DataLoader setup
def get_loader(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    return dataset, loader

# Initialize models
generator = Generator(latent_dim, style_dim, channels).to(device)
discriminator = Discriminator([3, 64, 128, 256, 512, 512]).to(device)  # Modified channel sizes

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.99))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.99))

# Training function
def train():
    _, dataloader = get_loader(image_size)  # Get the dataloader

    # Training loop
    for epoch in range(num_epochs):
        d_loss_avg = 0.0
        g_loss_avg = 0.0

        # Iterate over the dataset
        for real_images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train discriminator
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(z).detach()
            d_real = discriminator(real_images)
            d_fake = discriminator(fake_images)
            d_loss = torch.nn.functional.softplus(-d_real).mean() + torch.nn.functional.softplus(d_fake).mean()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_images = generator(z)
            g_loss = torch.nn.functional.softplus(-discriminator(fake_images)).mean()

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_loss_avg += d_loss.item()
            g_loss_avg += g_loss.item()

        print(f"Epoch {epoch+1}: D Loss: {d_loss_avg/len(dataloader):.4f}, G Loss: {g_loss_avg/len(dataloader):.4f}")

        # Save generated samples
        with torch.no_grad():
            z_sample = torch.randn(16, latent_dim, device=device)
            samples = generator(z_sample)
            save_samples(samples, epoch + 1, save_dir)

        # Optionally save model checkpoints
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs, or change as needed
            save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, epoch + 1, save_dir)

# Call the train function
if __name__ == "__main__":
    train()

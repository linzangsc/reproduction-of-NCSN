import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class CustomizedDataset:
    def __init__(self) -> None:
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.train_dataset = datasets.MNIST(root='../data', train=True, 
                                            download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root='../data', train=False, 
                                           download=True, transform=self.transform)

def visualize_float_result(image, axs):
    for i, img in enumerate(image):
        axs[i // 4, i % 4].imshow(img) 
        axs[i // 4, i % 4].axis('off') 
    return axs

def visualize_binary_result(image, output_path, row=4, col=4):
    fig, axs = plt.subplots(row, col, figsize=(8, 8))
    for i, img in enumerate(image):
        axs[i // 4, i % 4].imshow(img, cmap='binary') 
        axs[i // 4, i % 4].axis('off') 

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"visualization.jpg"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def visualize_latent_space(latents, labels, ax):
    latents = latents.cpu().numpy()
    labels = labels.cpu().numpy()
    ax.scatter(latents[:, 0], latents[:, 1], c=labels, s=10, cmap='hsv')
    return ax

class NoiseScheduler:
    def __init__(self, device, noise_step=10, min_std=0.01, max_std=1):
        super().__init__()
        self.device = device
        self.noise_step = noise_step
        self.noise_std = np.exp(np.linspace(np.log(min_std), 
                                            np.log(max_std), 
                                            num=noise_step), dtype=np.float32)

    def add_noise(self, x):
        batch_size = x.shape[0]
        noise = torch.randn_like(x).to(self.device)
        t = np.random.randint(0, self.noise_step, batch_size)
        sigma = self.noise_std[t]
        sigma = torch.from_numpy(sigma).to(self.device).view(-1, 1, 1, 1)
        noised_x = x + noise*sigma
        t = torch.tensor(t, dtype=torch.int32).to(self.device)
        return noised_x, sigma, t

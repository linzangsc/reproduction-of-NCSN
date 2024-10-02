import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import CustomizedDataset, visualize_float_result, visualize_latent_space, NoiseScheduler
from model import ScoreBasedModel
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, config) -> None:
        self.config = config
        self.image_size = config['image_size']
        self.logger = SummaryWriter(self.config['log_path'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.dataset = CustomizedDataset()
        self.train_dataset = self.dataset.train_dataset
        self.test_dataset = self.dataset.test_dataset
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                                        batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, 
                                                       batch_size=self.batch_size, shuffle=False)
        self.model = ScoreBasedModel(input_dim=config['input_dim'], output_dim=config['input_dim'],
                                     device=self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'], betas=(0.0, 0.999))
        self.noise_scheduler = NoiseScheduler(self.device)

    def loss(self, score, noised_x, images, sigma):
        # shape of inputs: (N, C, H, W)
        target = -1. / sigma**2 * (noised_x - images)
        loss = 0.5*((score - target)**2).sum(dim=(1,2,3)) * sigma.squeeze()**2
        return loss.mean()

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            for i, (images, _) in enumerate(self.train_loader):
                # translate to binary images
                images = images.to(self.device)
                self.optimizer.zero_grad()
                noised_x, sigma, t = self.noise_scheduler.add_noise(images)
                score = self.model(noised_x, t)
                loss = self.loss(score, noised_x, images, sigma)
                loss.backward()
                self.optimizer.step()
                if i % 1 == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], loss: {loss.item():.6f}')
                self.logger.add_scalar('loss/train', loss.item(), i + epoch * len(self.train_loader))

            self.save_model(self.config['ckpt_path'])
            with torch.no_grad():
                z = torch.rand((16, 1, self.image_size, self.image_size)).to(self.device)
                z = z * 2 - 1
                self.sample(z, epoch)

    def sample(self, z, epoch=-1, epsilon=2e-5):
        std_L = self.noise_scheduler.noise_std[0]
        x = z
        for i in range(self.noise_scheduler.noise_step-1, -1, -1):
            t = torch.ones((x.shape[0]), dtype=torch.int32).to(self.device) * i
            cur_std = self.noise_scheduler.noise_std[i]
            step_size = epsilon * cur_std**2/std_L**2
            x = self.model.sample_by_langevin(x, t, max_step=100, step_size=step_size)
        self.visualize_samples(x, epoch)

    def save_model(self, output_path):
        if not os.path.exists(output_path): os.mkdir(output_path)
        torch.save(self.model.state_dict(), os.path.join(output_path, f"model.pth"))

    def visualize_samples(self, sample_images, epoch):
        sample_images = sample_images.reshape(sample_images.shape[0], self.image_size, self.image_size).detach().to('cpu')
        npy_sampled_theta = np.array(sample_images)
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        axs = visualize_float_result(npy_sampled_theta, axs)
        self.logger.add_figure(f"sample results", plt.gcf(), epoch)
        plt.close(fig)

    def visualize_latent_space(self, epoch):
        fig, ax = plt.subplots()
        for (images, labels) in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, _, _, latents = self.model(images)
            ax = visualize_latent_space(latents, labels, ax)
        plt.colorbar(ax.collections[0], ax=ax)
        self.logger.add_figure(f"latent space", plt.gcf(), epoch)
        plt.close(fig)

if __name__ == "__main__":
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    trainer = Trainer(config=config)
    trainer.train()

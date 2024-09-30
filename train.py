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
                                                        batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, 
                                                       batch_size=self.batch_size, shuffle=False)
        self.model = ScoreBasedModel(input_dim=config['input_dim'], output_dim=config['input_dim'],
                                     device=self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'], betas=(0.0, 0.999))
        self.noise_scheduler = NoiseScheduler(self.device)

    def loss(self, score, all_noise):
        loss = 0.
        for i in range(self.noise_scheduler.noise_step):
            cur_score = score[i*self.batch_size:(i+1)*self.batch_size]
            cur_noise = all_noise[i*self.batch_size:(i+1)*self.batch_size]
            cur_std = self.noise_scheduler.noise_std[i]
            cur_lambda = cur_std**2
            cur_loss = 0.5*((cur_score + cur_noise/cur_std)**2).mean()
            loss += cur_lambda * cur_loss
        loss = loss / self.noise_scheduler.noise_step
        return loss

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # translate to binary images
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                all_noised_x, all_noise = self.noise_scheduler.add_noise(images)
                score = self.model(all_noised_x)
                loss = self.loss(score, all_noise)
                loss.backward()
                self.optimizer.step()
                if i % 1 == 0:
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], loss: {loss.item():.6f}')
                
            self.save_model(self.config['ckpt_path'])

            # z = torch.rand((16, 1, self.image_size, self.image_size)).to(self.device)
            # z = z * 2 - 1
            # sample_image = self.sampler.generate_samples(self.model, z, steps=256, step_size=10)
            # sample_image = sample_image.reshape(16, 1, self.image_size, self.image_size)
            # self.visualize_samples(sample_image, epoch)

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

import torch
import torch.nn as nn
    
class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, hidden_dim, num_embedding=10):
        super(ConditionalInstanceNorm2d, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(hidden_dim)
        self.time_embedding = nn.Embedding(num_embedding, hidden_dim)
    
    def forward(self, x, t):
        out = self.instance_norm(x)
        gamma = self.time_embedding(t)
        out = gamma*out
        return out

class LowBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LowBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)

class HighBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(HighBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim*2, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, x, skip_feature):
        x = torch.cat([skip_feature, x], dim=1)
        return self.net(x)

class ScoreBasedModel(nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_dim=64, denoise_step=10):
        super(ScoreBasedModel, self).__init__()
        self.device = device
        self.denoise_step = denoise_step
        self.time_embedding = nn.Embedding(denoise_step, 28*28)
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_dim*2, hidden_dim, kernel_size=3, padding=3),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.downlayer_1 = LowBlock(hidden_dim, hidden_dim)
        self.downsample_1 = nn.AvgPool2d(kernel_size=2)
        self.downlayer_2 = LowBlock(hidden_dim, hidden_dim*2)
        self.downsample_2 = nn.AvgPool2d(kernel_size=2)
        self.downlayer_3 = LowBlock(hidden_dim*2, hidden_dim*4)
        self.downsample_3 = nn.AvgPool2d(kernel_size=2)

        self.bottomlayer = LowBlock(hidden_dim*4, hidden_dim*8)

        self.uplayer_1 = HighBlock(hidden_dim*4, hidden_dim*4)
        self.upsample_1 = nn.ConvTranspose2d(hidden_dim*8, hidden_dim*4, kernel_size=3, 
                                             stride=2, padding=1, output_padding=1)
        self.uplayer_2 = HighBlock(hidden_dim*2, hidden_dim*2)
        self.upsample_2 = nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=3, 
                                             stride=2, padding=1, output_padding=1)
        self.uplayer_3 = HighBlock(hidden_dim, hidden_dim)
        self.upsample_3 = nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=3, 
                                             stride=2, padding=1, output_padding=1)
        self.output_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, output_dim, kernel_size=5),
        )
    
    def forward(self, x, t):
        # shape of x: (B, C, H, W)
        # time embedding
        t = self.time_embedding(t).reshape(x.shape)
        x = torch.cat([x, t], dim=1)
        # downsample
        x = self.input_layer(x)
        x1 = self.downlayer_1(x)
        x1_down = self.downsample_1(x1)
        x2 = self.downlayer_2(x1_down)
        x2_down = self.downsample_2(x2)
        x3 = self.downlayer_3(x2_down)
        x3_down = self.downsample_3(x3)
        # bottom layer
        x4 = self.bottomlayer(x3_down)
        # upsample
        x4_upsample = self.upsample_1(x4)
        x5 = self.uplayer_1(x4_upsample, x3)
        x5_upsample = self.upsample_2(x5)
        x6 = self.uplayer_2(x5_upsample, x2)
        x6_upsample = self.upsample_3(x6)
        x7 = self.uplayer_3(x6_upsample, x1)
        out = self.output_layer(x7)
        return out

    def sample_by_langevin(self, init_x, t, max_step=100, step_size=2e-5):
        x = init_x
        noise = torch.randn_like(x)
        for i in range(max_step):
            noise.normal_(0, 0.005)
            score = self(x, t)
            x = x + 0.5*step_size*score + torch.sqrt(torch.tensor(step_size))*noise
            x = x.clip(-1., 1.)
        return x.detach()

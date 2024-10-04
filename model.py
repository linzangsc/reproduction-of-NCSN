import torch
import torch.nn as nn
    
class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, hidden_dim, num_embedding=10):
        super(ConditionalInstanceNorm2d, self).__init__()
        self.hidden_dim = hidden_dim
        self.instance_norm = nn.InstanceNorm2d(hidden_dim, affine=False, track_running_stats=False)
        self.time_embedding = nn.Embedding(num_embedding, 2*hidden_dim)
        self.time_embedding.weight.data.normal_(0, 0.02)
    
    def forward(self, x, t):
        out = self.instance_norm(x)
        gamma, beta = self.time_embedding(t).chunk(chunks=2, dim=-1)
        out = gamma.view(-1, self.hidden_dim, 1, 1)*out + beta.view(-1, self.hidden_dim, 1, 1)
        return out

class LowBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LowBlock, self).__init__()
        self.norm_1 = ConditionalInstanceNorm2d(input_dim)
        self.conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm_2 = ConditionalInstanceNorm2d(hidden_dim)
        self.conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x, t):
        out = self.act(self.conv_1(self.norm_1(x, t)))
        out = self.act(self.conv_2(self.norm_2(out, t)))
        
        return out

class HighBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(HighBlock, self).__init__()
        self.norm_1 = ConditionalInstanceNorm2d(input_dim*2)
        self.conv_1 = nn.Conv2d(input_dim*2, hidden_dim, kernel_size=3, padding=1)
        self.norm_2 = ConditionalInstanceNorm2d(hidden_dim)
        self.conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x, skip_feature, t):
        x = torch.cat([skip_feature, x], dim=1)
        out = self.act(self.conv_1(self.norm_1(x, t)))
        out = self.act(self.conv_2(self.norm_2(out, t)))
        return out

class ScoreBasedModel(nn.Module):
    def __init__(self, input_dim, output_dim, device, hidden_dim=64, denoise_step=10):
        super(ScoreBasedModel, self).__init__()
        self.device = device
        self.denoise_step = denoise_step
        self.norm = ConditionalInstanceNorm2d(input_dim)
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=3),
        )
        self.downlayer_1 = LowBlock(hidden_dim, hidden_dim)
        self.downsample_1 = nn.AvgPool2d(kernel_size=2)
        self.downlayer_2 = LowBlock(hidden_dim, hidden_dim*2)
        self.downsample_2 = nn.AvgPool2d(kernel_size=2)
        self.downlayer_3 = LowBlock(hidden_dim*2, hidden_dim*4)
        self.downsample_3 = nn.AvgPool2d(kernel_size=2)
        self.downlayer_4 = LowBlock(hidden_dim*4, hidden_dim*8)
        self.downsample_4 = nn.AvgPool2d(kernel_size=2)

        self.bottomlayer = LowBlock(hidden_dim*8, hidden_dim*16)

        self.uplayer_0 = HighBlock(hidden_dim*8, hidden_dim*8)
        self.upsample_0 = nn.ConvTranspose2d(hidden_dim*16, hidden_dim*8, kernel_size=3, 
                                             stride=2, padding=1, output_padding=1)
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
        # shape of x: (B, C, H, W), shape of t: (B)
        # downsample
        x = self.input_layer(x)
        x1 = self.downlayer_1(x, t)
        x1_down = self.downsample_1(x1)
        x2 = self.downlayer_2(x1_down, t)
        x2_down = self.downsample_2(x2)
        x3 = self.downlayer_3(x2_down, t)
        x3_down = self.downsample_3(x3)
        x4 = self.downlayer_4(x3_down, t)
        x4_down = self.downsample_4(x4)
        # bottom layer
        x_bottom = self.bottomlayer(x4_down, t)
        # upsample
        x_bottom_upsample = self.upsample_0(x_bottom)
        x5 = self.uplayer_0(x_bottom_upsample, x4, t)
        x5_upsample = self.upsample_1(x5)
        x6 = self.uplayer_1(x5_upsample, x3, t)
        x6_upsample = self.upsample_2(x6)
        x7 = self.uplayer_2(x6_upsample, x2, t)
        x7_upsample = self.upsample_3(x7)
        x8 = self.uplayer_3(x7_upsample, x1, t)
        out = self.output_layer(x8)
        return out

    def sample_by_langevin(self, init_x, t, max_step=100, step_size=2e-5):
        x = init_x
        for i in range(max_step):
            noise = torch.randn_like(x)
            score = self(x, t)
            x = x + step_size*score + torch.sqrt(torch.tensor(2.*step_size))*noise
        return x.detach().clip(-1., 1.)

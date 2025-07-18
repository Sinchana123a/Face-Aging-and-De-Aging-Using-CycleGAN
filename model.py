import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        ]
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim*2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim*2),
                nn.ReLU(True),
            ]
            curr_dim *= 2
        for _ in range(n_residual_blocks):
            model += [ResnetBlock(curr_dim)]
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim//2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(curr_dim//2),
                nn.ReLU(True),
            ]
            curr_dim = curr_dim // 2
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        model = [
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        curr_dim = 64
        for i in range(3):
            model += [
                nn.Conv2d(curr_dim, curr_dim*2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim*2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            curr_dim *= 2
        model += [nn.Conv2d(curr_dim, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

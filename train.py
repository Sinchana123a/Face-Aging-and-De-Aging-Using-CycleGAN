import os
import itertools
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_dir = 'data/faceAging'
trainA = datasets.ImageFolder(root=data_dir, transform=transform)
trainB = datasets.ImageFolder(root=data_dir, transform=transform)
loaderA = DataLoader(trainA, batch_size=1, shuffle=True)
loaderB = DataLoader(trainB, batch_size=1, shuffle=True)

netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)
netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()

optimizer_G = optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(10):
    for i, (real_A, real_B) in enumerate(zip(loaderA, loaderB)):
        real_A = real_A[0].to(device)
        real_B = real_B[0].to(device)

        # Generator forward
        fake_B = netG_A2B(real_A)
        fake_A = netG_B2A(real_B)

        # Cycle loss
        rec_A = netG_B2A(fake_B)
        rec_B = netG_A2B(fake_A)

        # GAN loss
        loss_GAN_A2B = criterion_GAN(netD_B(fake_B), torch.ones_like(netD_B(fake_B)))
        loss_GAN_B2A = criterion_GAN(netD_A(fake_A), torch.ones_like(netD_A(fake_A)))

        # Total generator loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + 10 * (criterion_cycle(rec_A, real_A) + criterion_cycle(rec_B, real_B))

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Discriminator A
        loss_D_A = (criterion_GAN(netD_A(real_A), torch.ones_like(netD_A(real_A))) +
                    criterion_GAN(netD_A(fake_A.detach()), torch.zeros_like(netD_A(fake_A)))) * 0.5
        optimizer_D_A.zero_grad()
        loss_D_A.backward()
        optimizer_D_A.step()

        # Discriminator B
        loss_D_B = (criterion_GAN(netD_B(real_B), torch.ones_like(netD_B(real_B))) +
                    criterion_GAN(netD_B(fake_B.detach()), torch.zeros_like(netD_B(fake_B)))) * 0.5
        optimizer_D_B.zero_grad()
        loss_D_B.backward()
        optimizer_D_B.step()

    torch.save(netG_A2B.state_dict(), f'checkpoints/netG_A2B_epoch{epoch}.pth')
    torch.save(netG_B2A.state_dict(), f'checkpoints/netG_B2A_epoch{epoch}.pth')
    print(f'Epoch {epoch} completed.')

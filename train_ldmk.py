import argparse
import math
import os
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import wandb
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from tqdm import tqdm

from preprocess_ldmk import get_data_loader
from settings import DEVICE, IMAGE_SIZE, PARALLEL

cuda = True if torch.cuda.is_available() else False
wandb.init(project="papier_few_shot", entity="plop")


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = IMAGE_SIZE[0] // 4
        self.l1 = nn.Sequential(
            nn.Linear(256, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.InstanceNorm2d(128),
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(128, 128, 3, stride=1, padding=1)),
            nn.InstanceNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(128, 64, 3, stride=1, padding=1)),
            nn.InstanceNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 3, 3, stride=1, padding=1)),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.InstanceNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = IMAGE_SIZE[0] // 2 ** 4
        print(ds_size)
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        print(out.size())
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

generator = generator.to(DEVICE)
discriminator = discriminator.to(DEVICE)
adversarial_loss = adversarial_loss.to(DEVICE)

if PARALLEL:
    generator = nn.DataParallel(
        generator, device_ids=range(torch.cuda.device_count()))
    discriminator = nn.DataParallel(
        discriminator, device_ids=range(torch.cuda.device_count()))

try:
    if PARALLEL:
        generator.module.load_state_dict(torch.load(
            f"./weights/ldmk/generator_{IMAGE_SIZE[0]}.pt",
            map_location=DEVICE))
    else:
        generator.load_state_dict(torch.load(
            f"./weights/ldmk/generator_{IMAGE_SIZE[0]}.pt",
            map_location=DEVICE))
except RuntimeError:
    if PARALLEL:
        generator.module.load_state_dict(
            torch.load(f"./weights/ldmk/generator_{IMAGE_SIZE[0]}.bk",
                       map_location=DEVICE))
    else:
        generator.load_state_dict(
            torch.load(f"./weights/ldmk/generator_{IMAGE_SIZE[0]}.bk",
                       map_location=DEVICE))
except FileNotFoundError:
    generator.apply(weights_init_normal)
    print("\tFile not found, not loading weights generator...")

try:
    if PARALLEL:
        discriminator.module.load_state_dict(torch.load(
            f"./weights/ldmk/discriminator_{IMAGE_SIZE[0]}.pt",
            map_location=DEVICE))
    else:
        discriminator.load_state_dict(torch.load(
            f"./weights/ldmk/discriminator_{IMAGE_SIZE[0]}.pt",
            map_location=DEVICE))
except RuntimeError:
    if PARALLEL:
        discriminator.module.load_state_dict(
            torch.load(f"./weights/ldmk/discriminator_{IMAGE_SIZE[0]}.bk",
                       map_location=DEVICE))
    else:
        discriminator.load_state_dict(
            torch.load(f"./weights/ldmk/discriminator_{IMAGE_SIZE[0]}.bk",
                       map_location=DEVICE))
except FileNotFoundError:
    discriminator.apply(weights_init_normal)
    print("\tFile not found, not loading weights discriminator...")


# Initialize weights
wandb.watch((generator, discriminator))

# Configure data loader
dataloader = get_data_loader()
# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(999):
    for i, (imgs, _) in enumerate(tqdm(dataloader)):
        if cuda:
            imgs = imgs.cuda()
        # Adversarial ground truths
        print("Image size", imgs.size())
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0),
                         requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0),
                        requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 256))))

        # Generate a batch of images
        gen_imgs = generator(z)

        if i % 2 == 1:
            optimizer_G.zero_grad()
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # if (i % 2 == 0):
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()),
                                     fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(dataloader) + i
        wandb.log({"d_loss": d_loss.item(), "g_loss": g_loss.item()},
                  step=batches_done)
    if epoch % 10 == 0:
        images_to_grid = torch.cat((imgs, gen_imgs), dim=1).view(
            -1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])

        grid = torchvision.utils.make_grid(images_to_grid, padding=4, nrow=2,
                                           normalize=True, scale_each=True)

        wandb.log({"Img": [wandb.Image(grid, caption="image")]},
                  step=batches_done)

    if PARALLEL:
        torch.save(generator.module.state_dict(),
                   f"./weights/ldmk/generator_{IMAGE_SIZE[0]}.bk")
        torch.save(discriminator.module.state_dict(),
                   f"./weights/ldmk/discriminator_{IMAGE_SIZE[0]}.bk")
    else:
        torch.save(generator.state_dict(),
                   f"./weights/ldmk/generator_{IMAGE_SIZE[0]}.bk")
        torch.save(discriminator.state_dict(),
                   f"./weights/ldmk/discriminator_{IMAGE_SIZE[0]}.bk")
    copyfile(f"./weights/ldmk/discriminator_{IMAGE_SIZE[0]}.bk",
             f"./weights/ldmk/discriminator_{IMAGE_SIZE[0]}.pt")
    copyfile(f"./weights/ldmk/generator_{IMAGE_SIZE[0]}.bk",
             f"./weights/ldmk/generator_{IMAGE_SIZE[0]}.pt")

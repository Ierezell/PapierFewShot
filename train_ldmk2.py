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
from utils import weight_init, print_parameters
from preprocess import write_landmarks_on_image
from termcolor import colored
from preprocess_ldmk import get_data_loader
from settings import DEVICE, PARALLEL, IMAGE_SIZE
cuda = True if torch.cuda.is_available() else False
wandb.init(project="papier_few_shot", entity="plop")

print(colored(f"Device : {cuda}", "red"))
print(colored(f"Running on {torch.cuda.device_count()} GPUs.", "cyan"))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.drop = nn.Dropout(0.2)
        self.l1 = nn.Linear(100, 136*2)
        self.bn1 = nn.BatchNorm1d(136*2)

        self.l2 = nn.Linear(136*2, 136*2)
        self.bn2 = nn.BatchNorm1d(136*2)

        self.l3 = nn.Linear(136*2, 136)
        self.bn3 = nn.BatchNorm1d(136)

        self.relu = nn.SELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.l1(z)
        out = self.relu(out)
        out = self.drop(out)
        # out = self.bn1(out)

        out = self.l2(out)
        out = self.relu(out)
        out = self.drop(out)
        # out = self.bn2(out)

        out = self.l3(out)
        out = self.sigmoid(out)
        out = out.reshape((z.size(0), 68, 2))
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # self.drop = nn.Dropout(0.2)
        self.l1 = nn.Linear(68*2, 68*4)
        self.bn1 = nn.BatchNorm1d(68*4)

        self.l2 = nn.Linear(68*4, 68*2)
        self.bn2 = nn.BatchNorm1d(68*4)

        self.l3 = nn.Linear(68*2, 1)
        self.bn3 = nn.BatchNorm1d(68*2)
        self.l4 = nn.Linear(68*2, 1)

        self.relu = nn.SELU()
        self.drop = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = z.flatten(1)
        out = self.l1(z)
        out = self.relu(out)
        out = self.drop(out)
        # out = self.bn1(out)

        out = self.l2(out)
        out = self.relu(out)
        out = self.drop(out)
        # out = self.bn2(out)

        out = self.l3(out)
        out = self.relu(out)
        out = self.drop(out)
        # out = self.bn3(out)

        # out = self.l4(out)
        out = self.sigmoid(out)
        return out


if __name__ == "__main__":

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
                f"./weights/ldmk_int/generator_{IMAGE_SIZE[0]}.pt",
                map_location=DEVICE))
        else:
            generator.load_state_dict(torch.load(
                f"./weights/ldmk_int/generator_{IMAGE_SIZE[0]}.pt",
                map_location=DEVICE))
    except RuntimeError:
        if PARALLEL:
            generator.module.load_state_dict(
                torch.load(f"./weights/ldmk_int/generator_{IMAGE_SIZE[0]}.bk",
                           map_location=DEVICE))
        else:
            generator.load_state_dict(
                torch.load(f"./weights/ldmk_int/generator_{IMAGE_SIZE[0]}.bk",
                           map_location=DEVICE))
    except FileNotFoundError:
        generator.apply(weight_init)
        print("\tFile not found, not loading weights generator...")

    try:
        if PARALLEL:
            discriminator.module.load_state_dict(torch.load(
                f"./weights/ldmk_int/discriminator_{IMAGE_SIZE[0]}.pt",
                map_location=DEVICE))
        else:
            discriminator.load_state_dict(torch.load(
                f"./weights/ldmk_int/discriminator_{IMAGE_SIZE[0]}.pt",
                map_location=DEVICE))
    except RuntimeError:
        if PARALLEL:
            discriminator.module.load_state_dict(
                torch.load(f"./weights/ldmk_int/discriminator_{IMAGE_SIZE[0]}.bk",
                           map_location=DEVICE))
        else:
            discriminator.load_state_dict(
                torch.load(f"./weights/ldmk_int/discriminator_{IMAGE_SIZE[0]}.bk",
                           map_location=DEVICE))
    except FileNotFoundError:
        discriminator.apply(weight_init)
        print("\tFile not found, not loading weights discriminator...")

    # Initialize weights
    wandb.watch((generator, discriminator))

    print_parameters(generator)
    print_parameters(discriminator)
    # Configure data loader
    dataloader = get_data_loader()
    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=0.00002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    for epoch in range(99999):
        for i, ldmks in enumerate(tqdm(dataloader)):
            if cuda:
                ldmks = ldmks.cuda()
            ldmks = ldmks / 224
            # Adversarial ground truths
            # print(ldmks.size())
            valid = Variable(Tensor(ldmks.shape[0], 1).fill_(1.0),
                             requires_grad=False)
            fake = Variable(Tensor(ldmks.shape[0], 1).fill_(0.0),
                            requires_grad=False)

            # real_imgs = Variable(ldmks.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (ldmks.shape[0], 100))))
            # Generate a batch of images
            gen_ldmk = generator(z)
            # print(gen_ldmk[0][0])
            # print(ldmks[0][0])

            # if i % 2 == 0:
            optimizer_G.zero_grad()
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_ldmk), valid)
            # tqdm.write(f"{discriminator(gen_ldmk).mean()}")
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # if (i % 2 == 0):
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            # print(discriminator(real_imgs))
            real_loss = adversarial_loss(discriminator(ldmks), valid)
            # print(real_loss)
            fake_loss = adversarial_loss(discriminator(gen_ldmk.detach()),
                                         fake)
            # tqdm.write(f"{discriminator(ldmks).mean()}")
            # tqdm.write(f"{discriminator(gen_ldmk.detach()).mean()}")
            # print(discriminator(gen_ldmk.detach()))
            # print(fake_loss)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            batches_done = epoch * len(dataloader) + i
            # print("dloss :: ", d_loss)
            # print("gloss :: ", g_loss)
            wandb.log({"d_loss": d_loss.item(), "g_loss": g_loss.item()},
                      step=batches_done)
        if epoch % 100 == 0:
            gt_ldmk_im = []
            gen_ldmk_im = []
            for k in range(gen_ldmk.size(0)):
                gtli = np.zeros((224, 224, 3), np.float32)
                genli = np.zeros((224, 224, 3), np.float32)

                l_real = ldmks[k]*224
                l_gen = gen_ldmk[k]*224
                gtli = write_landmarks_on_image(gtli, l_real.int().cpu())
                genli = write_landmarks_on_image(genli, l_gen.int().cpu())
                gtli = transforms.ToTensor()(gtli)
                genli = transforms.ToTensor()(genli)

                gt_ldmk_im.append(gtli)
                gen_ldmk_im.append(genli)

            gt_ldmk_im = torch.stack(gt_ldmk_im)
            gen_ldmk_im = torch.stack(gen_ldmk_im)

            images_to_grid = torch.cat((gt_ldmk_im, gen_ldmk_im), dim=1).view(
                -1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])

            grid = torchvision.utils.make_grid(images_to_grid, padding=4, nrow=2,
                                               normalize=True, scale_each=True)

            wandb.log({"Img": [wandb.Image(grid, caption="image")]},
                      step=batches_done)

            if PARALLEL:
                torch.save(generator.module.state_dict(),
                           f"./weights/ldmk_int/generator_{IMAGE_SIZE[0]}.bk")
                torch.save(discriminator.module.state_dict(),
                           f"./weights/ldmk_int/discriminator_{IMAGE_SIZE[0]}.bk")
            else:
                torch.save(generator.state_dict(),
                           f"./weights/ldmk_int/generator_{IMAGE_SIZE[0]}.bk")
                torch.save(discriminator.state_dict(),
                           f"./weights/ldmk_int/discriminator_{IMAGE_SIZE[0]}.bk")
            copyfile(f"./weights/ldmk_int/discriminator_{IMAGE_SIZE[0]}.bk",
                     f"./weights/ldmk_int/discriminator_{IMAGE_SIZE[0]}.pt")
            copyfile(f"./weights/ldmk_int/generator_{IMAGE_SIZE[0]}.bk",
                     f"./weights/ldmk_int/generator_{IMAGE_SIZE[0]}.pt")

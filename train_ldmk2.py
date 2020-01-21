

from torch import nn
from shutil import copyfile
import platform
import sys

import torch
import torchvision
import wandb
from termcolor import colored
from torch.optim import SGD, Adam, RMSprop
from tqdm import tqdm, trange
from models_ldmk import Discriminator, Generator
from preprocess_ldmk import get_data_loader
from utils import load_losses
from settings import DEVICE, BATCH_SIZE, IN_DISC
from losses import adverserialLoss
from utils import weight_init, print_parameters
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.autograd.set_detect_anomaly(True)


class Checkpoints:
    def __init__(self, len_loader):
        self.losses = []
        self.best_loss = 1e10
        self.last_save = 0
        self.save_every = len_loader//4

    def save(self, loss, gen, disc):
        loss = loss.detach()
        self.last_save += 1
        if loss < self.best_loss or self.last_save > self.save_every:
            self.last_save = 0
            print('\n' + '-'*20)
            print("| Poids sauvegardes |")
            print('-'*20)
            self.best_loss = loss
            torch.save(gen.state_dict(),  "./weights/ldmk/gen.pt")
            torch.save(disc.state_dict(),  "./weights/ldmk/disc.pt")
            copyfile("./weights/ldmk/gen.pt", "./weights/ldmk/gen.bk")
            copyfile("./weights/ldmk/disc.pt", "./weights/ldmk/disc.bk")


if __name__ == '__main__':
    print(colored(f"Python : {sys.version}", 'blue'))
    print(colored(f"Torch version : {torch.__version__}", 'green'))
    print(colored(f"Torch CuDNN version : {torch.backends.cudnn.version()}",
                  'cyan'))
    print(colored(f"Device : {DEVICE}", "red"))
    print(colored(f"Running on {torch.cuda.device_count()} GPUs.", "cyan"))

    print(colored("Loading Dataset...", 'cyan'))

    train_loader, _ = get_data_loader()
    print(colored("Dataset Ok", "green"))
    print("Loading Models & Losses")
    print(colored("Loading Models", "cyan"))
    gen = Generator()
    disc = Discriminator()
    try:
        gen.load_state_dict(torch.load("./weights/ldmk/gen.pt",
                                       map_location=DEVICE))
    except RuntimeError:
        gen.load_state_dict(torch.load("./weights/ldmk/gen.bk",
                                       map_location=DEVICE))
    except FileNotFoundError:
        print("weights gen not found")
    try:
        disc.load_state_dict(torch.load("./weights/ldmk/disc.pt",
                                        map_location=DEVICE))
    except RuntimeError:
        disc.load_state_dict(torch.load("./weights/ldmk/disc.bk",
                                        map_location=DEVICE))
    except FileNotFoundError:
        print("weights disc not found")

    gen = gen.to(DEVICE)
    disc = disc.to(DEVICE)
    gen = gen.apply(weight_init)
    disc = disc.apply(weight_init)
    print_parameters(gen)
    print_parameters(disc)
    print(colored("Models Ok", "green"))
    print(colored("Loading Losses", "cyan"))
    advLoss, mchLoss, cntLoss, dscLoss = load_losses()
    print(colored("Losses Ok", "green"))

    optimizerGen = Adam(gen.parameters(), lr=0.0005)
    optimizerDisc = Adam(disc.parameters(), lr=0.0005)

    check = Checkpoints(len(train_loader))

    wandb.watch((gen, disc))
    advLoss = adverserialLoss()
    advLoss = advLoss.to(DEVICE)
    lossL1 = nn.L1Loss()
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    # ##########
    # Training #
    # ##########
    # torch.autograd.set_detect_anomaly(True)
    # TTUR = False
    # for i_epoch in trange(5):
    #     print("Epoch ! Epoch ! Epooooooch !!")
    #     for i_batch, batch in enumerate(tqdm(train_loader)):
    #         step = (i_epoch * len(train_loader)) + i_batch
    #         optimizerGen.zero_grad()
    #         gt_landmarks, itemIds = batch
    #         gt_landmarks = gt_landmarks.to(DEVICE)
    #         input_noise = torch.randn(BATCH_SIZE, 10, 1, 1, device=DEVICE)
    #         synth_im = gen(input_noise)
    #         loss_L1 = lossL1(synth_im, gt_landmarks)
    #         loss_L1.backward()
    #         optimizerGen.step()
    #         wandb.log({"lossL1": loss_L1}, step=step)
    #         # check.save(loss_fake, gen, disc)
    #         if i_batch % (len(train_loader)//4) == 0:
    #             images_to_grid = torch.cat((gt_landmarks, synth_im),
    #                                        dim=1).view(-1, 3, 224, 224)
    #             grid = torchvision.utils.make_grid(
    #                 images_to_grid, padding=4, nrow=2,
    #                 normalize=True, scale_each=True)
    #             wandb.log({"Img": [wandb.Image(grid, caption="image")]},
    #                       step=step)
    #             if platform.system() != "Windows":
    #                 wandb.save("./weights/ldmk/disc.pt")
    #                 wandb.save("./weights/ldmk/gen.pt")
    # BIG_STEP = step
    BIG_STEP = 0
    for i_epoch in trange(99):
        print("Epoch ! Epoch ! Epooooooch !!")

        for i_batch, batch in enumerate(tqdm(train_loader)):
            step = BIG_STEP + (i_epoch * len(train_loader)) + i_batch
            gt_landmarks, itemIds = batch
            gt_landmarks = gt_landmarks.to(DEVICE)
            itemIds = itemIds.to(DEVICE)
            input_noise = torch.randn(BATCH_SIZE, 10, 1, 1, device=DEVICE)
            synth_im = gen(input_noise)
            if ((i_batch % 10) == 0):
                # ####################
                # OPTI DISCRIMINATOR #
                # ####################
                optimizerDisc.zero_grad()

                score_gt = disc(
                    gt_landmarks + ((torch.randn_like(gt_landmarks)
                                     * gt_landmarks.max())/32))

                loss_real = criterion(score_gt,
                                      torch.ones_like(score_gt,
                                                      device=DEVICE))
                loss_real.backward()
                synth_im_noisy = synth_im + ((torch.randn_like(synth_im)
                                              * synth_im.max())/32)
                score_synth = disc(
                    synth_im_noisy.detach())

                loss_fake = criterion(score_synth,
                                      torch.zeros_like(score_synth,
                                                       device=DEVICE))
                loss_fake.backward()
                optimizerDisc.step()
                wandb.log({"score_synth": score_synth.mean(),
                           "score_gt": score_gt.mean(),
                           "loss_fake": loss_fake.mean(),
                           "loss_real": loss_real.mean()}, step=step)
            else:
                # ################
                # OPTI GENERATOR #
                # ################
                optimizerGen.zero_grad()
                score_gen = disc(synth_im)
                loss_gen = criterion(score_gen, torch.ones_like(score_gen,
                                                                device=DEVICE))
                loss_gen.backward()
                optimizerGen.step()
                wandb.log({"score_gen": score_gen.mean(),
                           "loss_gen": loss_gen.mean()}, step=step)
            # check.save(loss_fake, gen, disc)
            if i_batch % (len(train_loader)//4) == 0:
                images_to_grid = torch.cat((gt_landmarks, synth_im),
                                           dim=1).view(-1, 3, 224, 224)
                grid = torchvision.utils.make_grid(
                    images_to_grid, padding=4, nrow=2,
                    normalize=True, scale_each=True)
                wandb.log({"Img": [wandb.Image(grid, caption="image")]},
                          step=step)
                if platform.system() != "Windows":
                    wandb.save("./weights/ldmk/disc.pt")
                    wandb.save("./weights/ldmk/gen.pt")

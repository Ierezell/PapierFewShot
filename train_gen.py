

import platform
import sys

import torch
import torchvision
import wandb
from termcolor import colored, cprint
from torch import nn
from torch.optim import SGD, Adam
from torchvision import transforms
from tqdm import tqdm, trange

from preprocess import get_data_loader
from settings import (DEVICE, HALF, K_SHOT, LEARNING_RATE_DISC,
                      LEARNING_RATE_EMB, LEARNING_RATE_GEN, NB_EPOCHS,
                      PARALLEL, PATH_WEIGHTS_DISCRIMINATOR, BATCH_SIZE,
                      PATH_WEIGHTS_EMBEDDER, PATH_WEIGHTS_GENERATOR, TTUR)
from utils import (CheckpointsFewShots, load_losses, load_models, print_device,
                   print_parameters, weight_init)
# from losses import ldmkLoss
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':

    print(colored(f"Python : {sys.version}", 'blue'))
    print(colored(f"Torch version : {torch.__version__}", 'green'))
    print(colored(f"Torch CuDNN version : {torch.backends.cudnn.version()}",
                  'cyan'))
    print(colored(f"Device : {DEVICE}", "red"))

    print(colored("Loading Dataset...", 'cyan'))
    train_loader, nb_pers = get_data_loader()
    print(colored("Dataset Ok", "green"))

    print(colored("Loading Models", "cyan"))
    emb, gen, disc = load_models(nb_pers)
    print(colored("Models Ok", "green"))
    print(colored("Loading Losses", "cyan"))
    advLoss, mchLoss, cntLoss, dscLoss = load_losses()
    print(colored("Losses Ok", "green"))

    optimizerEmb = Adam(emb.parameters(), lr=LEARNING_RATE_EMB)
    optimizerGen = Adam(gen.parameters(), lr=LEARNING_RATE_GEN)

    len_loader = len(train_loader)
    check = CheckpointsFewShots(len_loader)

    print_parameters(emb)
    print_parameters(gen)
    print_parameters(cntLoss)

    print_device(emb)
    print_device(gen)
    print_device(cntLoss)

    wandb.watch((gen, emb))

    # ldmkLoss = ldmkLoss()
    # ldmkLoss = ldmkLoss.to(DEVICE)
    # ##########
    # Training #
    # ##########
    # torch.autograd.set_detect_anomaly(True)
    for i_epoch in trange(NB_EPOCHS):
        print("Epoch ! Epoch ! Epooooooch !!")
        for i_batch, batch in enumerate(tqdm(train_loader)):

            optimizerEmb.zero_grad()
            optimizerGen.zero_grad()

            gt_im, gt_landmarks, context, itemIds = batch

            gt_im = gt_im.to(DEVICE)
            gt_landmarks = gt_landmarks.to(DEVICE)
            context = context.to(DEVICE)
            itemIds = itemIds.to(DEVICE)

            embeddings, paramWeights, paramBias, layersUp = emb(context)
            # print(paramBias.size())
            # print(paramWeights.size())
            synth_im = gen(gt_landmarks, paramWeights, paramBias, layersUp)

            lossCnt = cntLoss(gt_im, synth_im)
            # lossLdmk = ldmkLoss(gt_im, synth_im)
            # loss = lossCnt
            # lossCnt.backward(torch.ones(
            #     torch.cuda.device_count(),
            #     dtype=(torch.half if HALF else torch.float),
            #     device=DEVICE))
            lossCnt.backward()
            optimizerEmb.step()
            optimizerGen.step()

            check.save("embGen", lossCnt, emb, gen, disc)

            wandb.log({"lossCnt": lossCnt.mean()},
                      #    "lossLdmk": lossLdmk.mean()},
                      step=(i_epoch*(len_loader*BATCH_SIZE))+i_batch)

            if i_batch % (len(train_loader)//2) == 0:

                images_to_grid = torch.cat((gt_landmarks, synth_im,
                                            gt_im, context),
                                           dim=1).view(-1, 3, 224, 224)

                grid = torchvision.utils.make_grid(
                    images_to_grid, padding=4, nrow=3 + 2*K_SHOT,
                    normalize=True, scale_each=True)

                wandb.log({"Img": [wandb.Image(grid.cpu(), caption="image")]},
                          step=(i_epoch*(len_loader*BATCH_SIZE))+i_batch)
                if platform.system() != "Windows":
                    wandb.save(PATH_WEIGHTS_EMBEDDER)
                    wandb.save(PATH_WEIGHTS_GENERATOR)
                    wandb.save(PATH_WEIGHTS_DISCRIMINATOR)

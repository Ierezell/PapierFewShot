

import platform
import sys

import torch
import torchvision
import wandb
from termcolor import colored
from torch.optim import SGD, Adam, RMSprop
from tqdm import tqdm, trange

from preprocess import get_data_loader
from settings import (DEVICE, HALF, IN_DISC, K_SHOT, LEARNING_RATE_DISC,
                      LEARNING_RATE_EMB, LEARNING_RATE_GEN, NB_EPOCHS,
                      PARALLEL, PATH_WEIGHTS_DISCRIMINATOR, BATCH_SIZE,
                      PATH_WEIGHTS_EMBEDDER, PATH_WEIGHTS_GENERATOR, TTUR)
from utils import (CheckpointsFewShots, load_losses, load_models, print_device,
                   print_parameters)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    print(colored(f"Python : {sys.version}", 'blue'))
    print(colored(f"Torch version : {torch.__version__}", 'green'))
    print(colored(f"Torch CuDNN version : {torch.backends.cudnn.version()}",
                  'cyan'))
    print(colored(f"Device : {DEVICE}", "red"))
    print(colored(f"Running on {torch.cuda.device_count()} GPUs.", "cyan"))

    print(colored("Loading Dataset...", 'cyan'))

    train_loader, nb_pers = get_data_loader()
    print(colored("Dataset Ok", "green"))
    print("Loading Models & Losses")
    print(colored("Loading Models", "cyan"))
    emb, gen, disc = load_models(nb_pers)
    print(colored("Models Ok", "green"))
    print(colored("Loading Losses", "cyan"))
    advLoss, mchLoss, cntLoss, dscLoss = load_losses()
    print(colored("Losses Ok", "green"))

    optimizerEmb = Adam(emb.parameters(), lr=LEARNING_RATE_EMB)
    optimizerGen = Adam(gen.parameters(), lr=LEARNING_RATE_GEN)
    optimizerDisc = Adam(disc.parameters(), lr=LEARNING_RATE_DISC)

    check = CheckpointsFewShots(len(train_loader))

    print_parameters(emb)
    print_parameters(gen)
    print_parameters(disc)
    print_parameters(advLoss)
    print_parameters(mchLoss)
    print_parameters(cntLoss)
    print_parameters(dscLoss)

    print_device(emb)
    print_device(gen)
    print_device(disc)
    print_device(advLoss)
    print_device(mchLoss)
    print_device(cntLoss)
    print_device(dscLoss)

    wandb.watch((gen, emb, disc))

    ones_grad = torch.ones(torch.cuda.device_count(),
                           dtype=(torch.half if HALF else torch.float),
                           device=DEVICE)
    # ##########
    # Training #
    # ##########
    # torch.autograd.set_detect_anomaly(True)
    for i_epoch in trange(NB_EPOCHS):
        print("Epoch ! Epoch ! Epooooooch !!")

        for i_batch, batch in enumerate(tqdm(train_loader)):
            step = (i_epoch * len(train_loader)) + i_batch

            gt_im, gt_landmarks, context, itemIds = batch

            gt_im = gt_im.to(DEVICE)
            gt_landmarks = gt_landmarks.to(DEVICE)
            context = context.to(DEVICE)
            itemIds = itemIds.to(DEVICE)

            embeddings, paramWeights, paramBias, layersUp = emb(context)
            synth_im = gen(gt_landmarks, paramWeights, paramBias, layersUp)

            optimizerEmb.zero_grad()
            optimizerGen.zero_grad()

            synth_im_w_ldmk = torch.cat((synth_im,
                                         gt_landmarks.detach()), dim=1)
            score_synth, feat_synth = disc(synth_im_w_ldmk, itemIds)

            gt_im_w_ldmk = torch.cat((gt_im,
                                      gt_landmarks.detach()), dim=1)
            score_gt, feat_gt = disc(gt_im_w_ldmk, itemIds)

            lossCnt = 10*cntLoss(gt_im, synth_im)
            lossAdv = advLoss(score_synth, feat_gt, feat_synth)
            if PARALLEL:
                lossMch = mchLoss(embeddings,
                                  disc.module.embeddings(itemIds).detach())
            else:
                lossMch = mchLoss(embeddings,
                                  disc.embeddings(itemIds).detach())

            loss = lossCnt + lossAdv + lossMch
            loss = loss.view(torch.cuda.device_count())
            ones_grad = torch.ones(torch.cuda.device_count(),
                                   dtype=(
                torch.half if HALF else torch.float),
                device=DEVICE)
            loss.backward(ones_grad)

            optimizerEmb.step()
            optimizerGen.step()

            if i_batch % 2 == 0:
                optimizerDisc.zero_grad()
                score_synth, feat_synth = disc(torch.cat((synth_im.detach(),
                                                          gt_landmarks.detach()), dim=1),
                                               itemIds)

                lossDsc = dscLoss(score_gt.detach(), score_synth)
                lossDsc = lossDsc.view(torch.cuda.device_count())
                ones_grad = torch.ones(torch.cuda.device_count(),
                                       dtype=(
                    torch.half if HALF else torch.float),
                    device=DEVICE)

                lossDsc.backward(ones_grad)
                optimizerDisc.step()

            check.save("embGen", loss.mean(), emb, gen, disc)
            check.save("disc", loss.mean(), emb, gen, disc)

            wandb.log({"Loss_dsc": lossDsc.mean()}, step=step)
            wandb.log({"lossCnt": lossCnt.mean()}, step=step)
            wandb.log({"lossMch": lossMch.mean()}, step=step)
            wandb.log({"lossAdv": lossAdv.mean()}, step=step)
            wandb.log({"LossTot": loss.mean()}, step=step)

            if i_batch % (len(train_loader)//4) == 0:
                images_to_grid = torch.cat((gt_landmarks, synth_im,
                                            gt_im, context),
                                           dim=1).view(-1, 3, 224, 224)
                grid = torchvision.utils.make_grid(
                    images_to_grid, padding=4, nrow=3 + 2*K_SHOT,
                    normalize=True, scale_each=True)
                wandb.log({"Img": [wandb.Image(grid, caption="image")]},
                          step=step)
                if platform.system() != "Windows":
                    wandb.save(PATH_WEIGHTS_EMBEDDER)
                    wandb.save(PATH_WEIGHTS_GENERATOR)
                    wandb.save(PATH_WEIGHTS_DISCRIMINATOR)
# if IN_DISC == "noisy":
#     gt_im = gt_im + ((torch.randn_like(gt_im)*gt_im.max())/32)

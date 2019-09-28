

from tqdm import tqdm, trange
import datetime
import os
import sys

import torch
import torchvision
import wandb
from torch.optim import SGD, Adam

from preprocess import get_data_loader
from settings import (CONFIG, DEVICE, K_SHOT, LEARNING_RATE_DISC,
                      LEARNING_RATE_EMB, LEARNING_RATE_GEN, LOAD_PREVIOUS,
                      NB_EPOCHS, PATH_WEIGHTS_DISCRIMINATOR,
                      PATH_WEIGHTS_EMBEDDER, PATH_WEIGHTS_GENERATOR,
                      PRINT_EVERY, TTUR)
from utils import (CheckpointsFewShots, load_losses, load_models, print_device,
                   print_parameters)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
date = datetime.datetime.now().replace(microsecond=0)
train_id = "_".join(CONFIG.values())

# os.environ['WANDB_MODE'] = 'dryrun'

wandb.init(project="papierfewshot",
           id=train_id,
           name=train_id,
           resume=LOAD_PREVIOUS,
           config=CONFIG)


if __name__ == '__main__':

    print("Python : ", sys.version)
    print("torch version : ", torch.__version__)
    print("Device : ", DEVICE)

    print("Loading Dataset")
    train_loader, nb_pers = get_data_loader()

    print("Loading Models & Losses")

    emb, gen, disc = load_models(nb_pers)
    advLoss, mchLoss, cntLoss, dscLoss = load_losses()

    optimizerEmb = Adam(emb.parameters(), lr=LEARNING_RATE_EMB)
    optimizerGen = Adam(gen.parameters(), lr=LEARNING_RATE_GEN)
    optimizerDisc = SGD(disc.parameters(), lr=LEARNING_RATE_DISC)

    check = CheckpointsFewShots()

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

    # ##########
    # Training #
    # ##########
    # torch.autograd.set_detect_anomaly(True)
    for i_epoch in trange(NB_EPOCHS):
        print("Epoch ! Epoch ! Epooooooch !!")
        for i_batch, batch in enumerate(tqdm(train_loader)):

            optimizerEmb.zero_grad()
            optimizerDisc.zero_grad()
            optimizerGen.zero_grad()

            gt_im, gt_landmarks, context, itemIds = batch
            # print("batch loade")

            gt_im = gt_im.to(DEVICE)
            gt_landmarks = gt_landmarks.to(DEVICE)
            context = context.to(DEVICE)
            itemIds = itemIds.to(DEVICE)

            embeddings, paramWeights, paramBias, layersUp = emb(context)
            # print("emb ok")
            synth_im = gen(gt_landmarks,  paramWeights, paramBias, layersUp)
            # print("gen ok")

            score_synth, feature_maps_disc_synth = disc(torch.cat(
                (synth_im, gt_landmarks), dim=1), itemIds)
            gt_w_ldm = torch.cat((gt_im, gt_landmarks), dim=1)
            score_gt, feature_maps_disc_gt = disc(
                gt_w_ldm+(torch.randn_like(gt_w_ldm)/2), itemIds)
            # print("disc ok")

            lossDsc = dscLoss(score_gt, score_synth).mean()
            lossAdv = advLoss(score_synth, feature_maps_disc_gt,
                              feature_maps_disc_synth).mean()
            lossCnt = cntLoss(gt_im, synth_im).mean()
            lossMch = mchLoss(embeddings,
                              disc.module.embeddings(itemIds)).mean()

            loss = lossAdv + lossCnt + lossMch

            # print("loss ok")
            if TTUR:
                if i_batch % 3 == 0:
                    lossDsc.backward(torch.ones(torch.cuda.device_count(),
                                                device=DEVICE))
                    optimizerDisc.step()
                else:
                    loss.backward(torch.ones(torch.cuda.device_count(),
                                             device=DEVICE))
                    optimizerEmb.step()
                    optimizerGen.step()
            else:
                loss = loss + lossDsc
                loss.backward(torch.ones(torch.cuda.device_count(),
                                         device=DEVICE))

                optimizerDisc.step()
                optimizerEmb.step()
                optimizerGen.step()

            check.save("embGen", loss, emb, gen, disc)
            check.save("disc", lossDsc, emb, gen, disc)

            wandb.log({"Loss_dsc": lossDsc})
            wandb.log({"lossCnt": lossCnt})
            wandb.log({"lossMch": lossMch})
            wandb.log({"lossAdv": lossAdv})
            wandb.log({"LossTot": loss})

            if i_batch % PRINT_EVERY == 0:  # and i_batch != 0:
                images_to_grid = torch.cat((gt_landmarks, synth_im,
                                            gt_im, context),
                                           dim=1).view(-1, 3, 224, 224)

                grid = torchvision.utils.make_grid(
                    images_to_grid, padding=4, nrow=3 + K_SHOT,
                    normalize=True, scale_each=True)

                wandb.log({"Img": [wandb.Image(grid, caption="image")]})
                # wandb.save(PATH_WEIGHTS_EMBEDDER)
                # wandb.save(PATH_WEIGHTS_GENERATOR)
                # wandb.save(PATH_WEIGHTS_DISCRIMINATOR)

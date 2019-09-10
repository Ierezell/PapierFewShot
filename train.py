

import sys
import torch
import torchvision
from torch.optim import Adam, SGD

from preprocess import get_data_loader
from settings import (DEVICE, K_SHOT, LEARNING_RATE_DISC, LEARNING_RATE_EMB,
                      LEARNING_RATE_GEN, NB_EPOCHS, PRINT_EVERY, CONFIG,
                      LOAD_PREVIOUS, PATH_WEIGHTS_EMBEDDER,
                      PATH_WEIGHTS_GENERATOR, PATH_WEIGHTS_DISCRIMINATOR,
                      )
from utils import (CheckpointsFewShots, load_losses, load_models,
                   print_parameters)
import datetime
import wandb
wandb.init(project="papierfewshot",
           name=f"test-{datetime.datetime.now().replace(microsecond=0)}",
           resume=LOAD_PREVIOUS,
           config=CONFIG)

print("Python : ", sys.version)
print("torch version : ", torch.__version__)
print("Device : ", DEVICE)

train_loader, nb_pers = get_data_loader()

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

wandb.watch((gen, emb, disc))

# ##########
# Training #
# ##########
# torch.autograd.set_detect_anomaly(True)

for i_epoch in range(NB_EPOCHS):
    print("Epoch ! Epoch ! Epooooooch !!")
    for i_batch, batch in enumerate(train_loader):

        optimizerEmb.zero_grad()
        optimizerDisc.zero_grad()
        optimizerGen.zero_grad()

        gt_im, gt_landmarks, context, itemIds = batch

        gt_im = gt_im.to(DEVICE)
        gt_landmarks = gt_landmarks.to(DEVICE)
        context = context.to(DEVICE)
        itemIds = itemIds.to(DEVICE)

        embeddings, paramWeights, paramBias, layersUp = emb(context)
        synth_im = gen(gt_landmarks,  paramWeights, paramBias, layersUp)

        score_synth, feature_maps_disc_synth = disc(torch.cat((synth_im,
                                                               gt_landmarks),
                                                              dim=1), itemIds)
        gt_w_ldm = torch.cat((gt_im, gt_landmarks), dim=1)
        score_gt, feature_maps_disc_gt = disc(
            gt_w_ldm+(torch.randn_like(gt_w_ldm)/2), itemIds)

        lossDsc = dscLoss(score_gt, score_synth)
        lossDsc = lossDsc.mean()
        lossAdv = advLoss(score_synth, feature_maps_disc_gt,
                          feature_maps_disc_synth)
        lossCnt = cntLoss(gt_im, synth_im)
        lossMch = mchLoss(embeddings, disc.module.embeddings(itemIds))
        loss = lossAdv + lossCnt + lossMch
        loss = loss.mean()

        loss_totale = loss + lossDsc
        loss_totale.backward(torch.cuda.FloatTensor(
            torch.cuda.device_count()).fill_(1))

        optimizerDisc.step()
        optimizerEmb.step()
        optimizerGen.step()

        check.addCheckpoint("cnt", torch.sum(lossCnt, dim=-1))
        check.addCheckpoint("adv", torch.sum(lossAdv, dim=-1))
        check.addCheckpoint("mch", torch.sum(lossMch, dim=-1))
        check.addCheckpoint("dsc", torch.sum(lossDsc, dim=-1))

        check.save("embGen", torch.sum(loss, dim=-1), emb, gen, disc)
        check.save("disc", torch.sum(lossDsc, dim=-1), emb, gen, disc)

        wandb.log({"Loss_dsc": torch.sum(lossDsc, dim=-1)})
        wandb.log({"lossCnt": torch.sum(lossCnt, dim=-1)})
        wandb.log({"lossMch": torch.sum(lossMch, dim=-1)})
        wandb.log({"lossAdv": torch.sum(lossAdv, dim=-1)})
        wandb.log({"LossTot": torch.sum(loss, dim=-1)})

        if i_batch % PRINT_EVERY == 0 and i_batch != 0:

            images_to_grid = torch.cat((gt_landmarks, synth_im,
                                        gt_im, context),
                                       dim=1).view(-1, 3, 224, 224)

            grid = torchvision.utils.make_grid(
                images_to_grid, padding=4, nrow=3 + K_SHOT,
                normalize=True, scale_each=True)

            wandb.log({"Img": [wandb.Image(grid, caption="image")]})
            wandb.save(PATH_WEIGHTS_EMBEDDER)
            wandb.save(PATH_WEIGHTS_GENERATOR)
            wandb.save(PATH_WEIGHTS_DISCRIMINATOR)

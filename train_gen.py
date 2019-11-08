

from tqdm import tqdm, trange
import sys

import torch
import torchvision
import wandb
from torchvision import transforms
from torch.optim import SGD, Adam

from preprocess import get_data_loader
from settings import (DEVICE, K_SHOT, LEARNING_RATE_DISC, LEARNING_RATE_EMB,
                      LEARNING_RATE_GEN, NB_EPOCHS, PRINT_EVERY, TTUR,
                      PATH_WEIGHTS_EMBEDDER, PATH_WEIGHTS_GENERATOR,
                      PATH_WEIGHTS_DISCRIMINATOR, HALF)
from utils import (CheckpointsFewShots, load_losses, load_models, print_device,
                   print_parameters)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

if __name__ == '__main__':

    print("Python : ", sys.version)
    print("Torch version : ", torch.__version__)
    print("Torch CuDNN version : ", torch.backends.cudnn.version())
    print("Device : ", DEVICE)

    print("Loading Dataset")
    train_loader, nb_pers = get_data_loader()

    print("Loading Models & Losses")
    emb, gen, disc = load_models(nb_pers)
    advLoss, mchLoss, cntLoss, dscLoss = load_losses()

    optimizerEmb = Adam(emb.parameters(), lr=LEARNING_RATE_EMB)
    optimizerGen = Adam(gen.parameters(), lr=LEARNING_RATE_GEN)

    check = CheckpointsFewShots()

    print_parameters(emb)
    print_parameters(gen)
    print_parameters(cntLoss)

    print_device(emb)
    print_device(gen)
    print_device(cntLoss)

    wandb.watch((gen, emb))
    l1 = torch.nn.L1Loss()
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
            synth_im = gen(gt_landmarks, paramWeights, paramBias, layersUp)

            lossCnt = cntLoss(gt_im, synth_im).mean()
            lossL1 = l1(gt_im, synth_im).mean()*120
            loss = lossCnt + lossL1
            # loss = lossCnt

            # loss.backward(torch.ones(
            #     torch.cuda.device_count(),
            #     dtype=(torch.half if HALF else torch.float),
            #     device=DEVICE))
            loss.backward()
            optimizerEmb.step()
            optimizerGen.step()

            check.save("embGen", loss, emb, gen, disc)

            wandb.log({"loss": loss})
            wandb.log({"lossCnt": lossCnt})
            wandb.log({"lossL1": lossL1})

            if i_batch % PRINT_EVERY == 0:
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

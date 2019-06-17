

from torch.utils.tensorboard import SummaryWriter
from utils import (Checkpoints, load_models, plot_grad_flow, get_size)
from settings import (BATCH_SIZE, DEVICE, K_SHOT, LEARNING_RATE_EMB,
                      LEARNING_RATE_GEN, LEARNING_RATE_DISC, NB_EPOCHS,
                      PRINT_EVERY, LOAD_PREVIOUS, NB_WORKERS)
from preprocess import get_data_loader, frameLoader
from losses import adverserialLoss, contentLoss, discriminatorLoss, matchLoss
from archi import Discriminator, Embedder, Generator
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch
import numpy as np
import matplotlib.style as mplstyle
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("tkagg")

mplstyle.use(['dark_background', 'fast'])

# ######
# Init #
# ######


# class testLoader(Dataset):

#     def __init__(self):
#         super(testLoader, self).__init__()
#         self.plop = 3

#     def __getitem__(self, index):
#         a = torch.rand(3, 224, 224)
#         b = torch.rand(3, 224, 224)
#         c = torch.rand(K_SHOT*3, 224, 224)
#         d = torch.randint(0, 118, (1,))
#         return a, b, c, d

#     def __len__(self):
#         return 36237


# datas = frameLoader()
# train_loader = DataLoader(datas, batch_size=BATCH_SIZE,
#                           shuffle=True, num_workers=0, pin_memory=True)
# nb_pers = 118

train_loader, nb_pers = get_data_loader(K_shots=K_SHOT, workers=NB_WORKERS)

emb, gen, disc = load_models(nb_pers, load_previous_state=LOAD_PREVIOUS)
emb = emb.to(DEVICE)
gen = gen.to(DEVICE)
disc = disc.to(DEVICE)

print("Nombre de paramètres Emb: ",
      f"{sum([np.prod(p.size()) for p in emb.parameters()]):,}")

print("Nombre de paramètres Gen: ",
      f"{sum([np.prod(p.size()) for p in gen.parameters()]):,}")

print("Nombre de paramètres Disc: ",
      f"{sum([np.prod(p.size()) for p in disc.parameters()]):,}")


advLoss = adverserialLoss()
mchLoss = matchLoss()
cntLoss = contentLoss()
dscLoss = discriminatorLoss()

advLoss = advLoss.to(DEVICE)
mchLoss = mchLoss.to(DEVICE)
cntLoss = cntLoss.to(DEVICE)
dscLoss = dscLoss.to(DEVICE)

check = Checkpoints()
optimizerEmb = Adam(emb.parameters(), lr=LEARNING_RATE_EMB)
optimizerGen = Adam(gen.parameters(), lr=LEARNING_RATE_GEN)
optimizerDisc = Adam(disc.parameters(), lr=LEARNING_RATE_DISC)

plt.ion()
fig, axes = plt.subplots(3, 3, figsize=(15, 10), num='Mon')

# ##########
# Training #
# ##########
print("torch version : ", torch.__version__)
print("Device : ", DEVICE)
# torch.autograd.set_detect_anomaly(True)
for i_epoch in range(NB_EPOCHS):
    for i_batch, batch in enumerate(train_loader):
        # Passe avant #
        optimizerEmb.zero_grad()
        optimizerDisc.zero_grad()
        optimizerGen.zero_grad()
        print("test : ", i_epoch, i_batch)
        gt_im, gt_landmarks, context, itemIds = batch
        gt_im = gt_im.to(DEVICE)
        gt_landmarks = gt_landmarks.to(DEVICE)
        context = context.to(DEVICE)
        itemIds = itemIds.to(DEVICE)
        # print(gt_im.size())
        # print(gt_landmarks.size())
        # print(context.size())
        # print(itemIds.size())
        # print(gt_im.type(), gt_im.requires_grad)
        # print(gt_landmarks.type(), gt_landmarks.requires_grad)
        # print(context.type(), context.requires_grad)
        # print(itemIds.type(), itemIds.requires_grad)
        embeddings, paramNorm = emb(context)
        # print("1  ", embeddings.type())

        synth_im = gen(gt_landmarks, paramNorm)
        # print("Swaggggg")
        # print(synth_im.max())
        # print(gt_im.max())
        # print("2  ", synth_im.type())

        score_synth, feature_maps_disc_synth = disc(torch.cat((synth_im,
                                                               gt_landmarks),
                                                              dim=1), itemIds)
        # print("ALLLLLLLLLLA :  ", score_synth)

        score_gt, feature_maps_disc_gt = disc(torch.cat((gt_im, gt_landmarks),
                                                        dim=1), itemIds)
        # print("3  ", score_gt.type(), feature_maps_disc_gt[0].type())

        # #################
        # Calcul des loss #
        # #################
        if i_batch % 3 == 0 or i_batch % 3 == 1:
            lossDsc = dscLoss(score_gt, score_synth)
            lossDsc.backward()
            optimizerDisc.step()
            check.addCheckpoint("dsc", lossDsc)
            check.save("disc", lossDsc, emb, gen, disc)
        else:
            lossAdv = advLoss(score_synth, feature_maps_disc_gt,
                              feature_maps_disc_synth)
            lossCnt = (1e-4)*cntLoss(gt_im, synth_im)
            lossMch = 80*mchLoss(embeddings, disc.embeddings(itemIds))
            loss = lossAdv + lossCnt + lossMch
            loss.backward()
            optimizerEmb.step()
            optimizerGen.step()
            check.addCheckpoint("cnt", lossCnt)
            check.addCheckpoint("adv", lossAdv)
            check.addCheckpoint("mch", lossMch)
            check.save("embGen", loss, emb, gen, disc)

        # print("lossAdv",  lossAdv.data)
        # print("lossCnt",  lossCnt.data)
        # print("lossMch",  lossMch.data)
        # print("lossDsc",  lossDsc.data, score_gt.data, score_synth.data)
        if i_batch % PRINT_EVERY == 0:
            # writer = SummaryWriter()
            # grid = torchvision.utils.make_grid(
            #     torch.cat((gt_im, gt_landmarks, context),
            #               dim=1).view(-1, 3, 224, 224))
            # writer.add_image('images', grid, 0)
            # writer.add_graph(emb, context)
            # writer.add_graph(disc,
            #                  (torch.cat((synth_im, gt_landmarks), dim=1),
            #                   itemIds))
            # writer.add_graph(gen, gt_landmarks)
            # writer.close()
            check.visualize(fig, axes,
                            gt_landmarks,
                            synth_im,
                            gt_im,
                            emb, gen, disc)

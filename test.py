

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from archi import Discriminator, Embedder, Generator
from losses import adverserialLoss, contentLoss, discriminatorLoss, matchLoss
from preprocess import get_data_loader, frameLoader
from settings import BATCH_SIZE, DEVICE, K_SHOT, LEARNING_RATE, NB_EPOCHS
from utils import (Checkpoints, load_models, plot_grad_flow, get_size)

mplstyle.use(['dark_background', 'fast'])

# ######
# Init #
# ######


class testLoader(Dataset):

    def __init__(self):
        super(testLoader, self).__init__()
        self.plop = 3

    def __getitem__(self, index):
        a = torch.rand(3, 224, 224)
        b = torch.rand(3, 224, 224)
        c = torch.rand(K_SHOT*3, 224, 224)
        d = torch.randint(0, 118, (1,))
        return a, b, c, d

    def __len__(self):
        return 36237


datas = frameLoader()
train_loader = DataLoader(datas, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
nb_pers = 118

# train_loader, nb_pers = get_data_loader()

emb, gen, disc = load_models(nb_pers, load_previous_state=False)
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

check = Checkpoints()
optimizerEmb = Adam(emb.parameters(), lr=LEARNING_RATE)
optimizerGen = Adam(gen.parameters(), lr=LEARNING_RATE)
optimizerDisc = Adam(disc.parameters(), lr=LEARNING_RATE)

plt.ion()
fig, axes = plt.subplots(3, 3, figsize=(15, 10), num='Mon')
# ##########
# Training #
# ##########
print("torch version : ", torch.__version__)
# torch.autograd.set_detect_anomaly(True)
for i_epoch in range(NB_EPOCHS):
    for i_batch, batch in enumerate(train_loader):
        # Passe avant #
        optimizerEmb.zero_grad()
        optimizerDisc.zero_grad()
        optimizerGen.zero_grad()
        print("test : ", i_epoch, i_batch)
        gt_im, gt_landmarks, context, itemIds = batch
        gt_im.to(DEVICE)
        gt_landmarks.to(DEVICE)
        context.to(DEVICE)
        itemIds.to(DEVICE)
        print(gt_im.size())
        print(gt_landmarks.size())
        print(context.size())
        print(itemIds.size())
        print(gt_im.type(), gt_im.requires_grad)
        print(gt_landmarks.type(), gt_landmarks.requires_grad)
        print(context.type(), context.requires_grad)
        print(itemIds.type(), itemIds.requires_grad)
        embeddings = emb(context)
        # print("1  ", embeddings.requires_grad)

        synth_im = gen(gt_landmarks)
        score_synth, feature_maps_disc_synth = disc(torch.cat((synth_im,
                                                               gt_landmarks),
                                                              dim=1), itemIds)
        score_gt, feature_maps_disc_gt = disc(torch.cat((gt_im, gt_landmarks),
                                                        dim=1), itemIds)
        # #################
        # Calcul des loss #
        # #################
        lossAdv = advLoss(score_synth, feature_maps_disc_gt,
                          feature_maps_disc_synth)
        lossCnt = cntLoss(gt_im, synth_im)
        lossMch = mchLoss(embeddings, disc.embeddings(itemIds))
        lossDsc = dscLoss(score_gt, score_synth)
        print("lossAdv",  lossAdv.data)
        print("lossCnt",  lossCnt.data)
        print("lossMch",  lossMch.data)
        print("lossDsc",  lossDsc.data, score_gt.data, score_synth.data)
        loss = lossAdv + lossCnt + lossMch + lossDsc
        print("LOSS  : ", loss.requires_grad)
        loss.backward()
        optimizerEmb.step()
        optimizerDisc.step()
        optimizerGen.step()

        check.addCheckpoint(lossAdv + lossCnt + lossMch, lossDsc)
        check.save(lossAdv + lossCnt + lossMch, lossDsc, emb, gen, disc)
        if i_batch % 10 == 0:
            check.visualize(fig, axes,
                            gt_landmarks.detach()[0], synth_im.detach()[0],
                            gt_im.detach()[0], emb, gen, disc)

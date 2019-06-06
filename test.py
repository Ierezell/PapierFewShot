from settings import NB_EPOCHS, LEARNING_RATE
from preprocess import get_data_loader
from archi import Embedder, Generator, Discriminator
from losses import adverserialLoss, matchLoss, contentLoss, discriminatorLoss
from torch.optim import Adam
import torch
# ######
# Init #
# ######
train_loader, nb_pers = get_data_loader()
emb = Embedder()
gen = Generator()
disc = Discriminator(nb_pers)
advLoss = adverserialLoss()
mchLoss = matchLoss()
cntLoss = contentLoss()
dscLoss = discriminatorLoss()
optimizerEmb = Adam(emb.parameters(), lr=LEARNING_RATE)
optimizerGen = Adam(gen.parameters(), lr=LEARNING_RATE)
optimizerDisc = Adam(disc.parameters(), lr=LEARNING_RATE)
for i_epoch in range(NB_EPOCHS):
    for i_batch, batch in enumerate(train_loader):
        # #############
        # Passe avant #
        # #############
        print("test : ", i_epoch, i_batch)
        gt_im, gt_landmarks, context_tensors, itemIds = batch
        # print("itemIds", itemIds)
        # print("Gt_imsize", gt_im.size())
        out_emb, out_paramIn = emb(context_tensors)
        # print("emb size", out_emb.size())
        out_gen = gen(gt_landmarks, out_paramIn)
        out_gen = out_gen[:, :, 0:224, 0:224]
        # print("Gen size", out_gen.size())
        score_synth, feature_maps_synth = disc(torch.cat((out_gen,
                                                          gt_landmarks),
                                                         dim=1),
                                               itemIds)
        # print("Disc synth ok")
        score_gt, feature_maps_gt = disc(torch.cat((gt_im, gt_landmarks),
                                                   dim=1), itemIds)
        # #################
        # Calcul des loss #
        # #################
        print(score_synth)
        lossAdv = advLoss(score_synth, feature_maps_synth,
                          feature_maps_gt)
        # print("Adv loss ok ")
        lossCnt = cntLoss(gt_im, out_gen)
        # print("Cnt loss ok ")
        lossMch = mchLoss(out_emb, disc.embeddings(itemIds))
        # print("Mch loss ok ")
        lossDsc = dscLoss(score_gt, score_synth)
        print("Dsc loss ok ")
        print("lossAdv",  lossAdv)
        print("lossCnt",  lossCnt)
        print("lossMch",  lossMch)
        print("lossDsc",  lossDsc)
        loss = lossAdv + lossCnt + lossMch + lossDsc
        loss.backward()
        optimizerEmb.step()
        optimizerDisc.step()
        optimizerGen.step()

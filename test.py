from settings import NB_EPOCHS
from preprocess import get_data_loader
from Archi import Embedder, Generator, Discriminator
import torch
train_loader, nb_pers = get_data_loader()
emb = Embedder()
gen = Generator()
disc = Discriminator(nb_pers)
for i_epoch in range(NB_EPOCHS):
    for i_batch, batch in enumerate(train_loader):
        print("test : ", i_epoch, i_batch)
        gt_im, gt_landmarks, context_tensors, itemIds = batch
        print("itemIds", itemIds)
        print("Gt_imsize", gt_im.size())
        out_emb = emb(context_tensors)
        print("emb size", out_emb.size())
        out_gen = gen(gt_landmarks)[:, :, 0:224, 0:224]
        print("Gen size", out_gen.size())
        score, feature_maps = disc(torch.cat((out_gen, gt_landmarks), dim=1),
                                   itemIds)
        print(score)

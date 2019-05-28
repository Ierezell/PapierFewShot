"""
L'indice i dans le papier correspond à la i-ème video du dataset
l'indice t dans le papier correspond à la t-ème image de la i-ème vidéo

"""

import torch

from Archi import Discriminator, Embedder, Generator
from preprocess import get_landmarks
from settings import BATCH_SIZE, DEVICE, IMAGE_SIZE, LEARNING_RATE, NB_EPOCHS
from utils import load_loaders, load_models

embedder, generator, discriminator = load_models()
train_loader, valid_loader, test_loader = load_loaders()
optimizer =

for i_epoch in range(NB_EPOCHS):
    for i_batch, batch in enumerate(train_loader):
        K_images_landmarks, landmarks_gen, gt_img, gt_landmarks = batch

        K_images_landmarks = K_images_landmarks.to(DEVICE)
        landmarks_gen = landmarks_gen.to(DEVICE)
        gt_img = gt_img.to(DEVICE)
        gt_landmarks = gt_landmarks.to(DEVICE)

        optimizer.zero_grad()

        embedding = embedder(K_images_landmarks)
        synth_img = generator(landmarks_gen, embedding)
        score = discriminator(synth_img, get_landmarks(synth_img))

        lossEmbGen = lossContent(gt_img, synth_img) +\
            lossAdversarial(score) + lossMatching(ei, wi)
        lossDisc = lossDiscriminator(gt_img, synth_img)

        checkpoint(i_batch, lossEmbGen, lossDisc)

        lossEmbGen.backward()
        lossDisc.backward()
        optimizerEmbGen.step()
        optimizerDisc.step()

    # sheduler.step()

torch.save(Network.state_dict(), ROOT_WEIGHTS+'/BACKUP_GOOD_LFWTRIPLET.pt')

torch.save(Network.encoder.state_dict(),
           ROOT_WEIGHTS+'/BACKUP_GOOD_LFWTRIPLETENCODER.pt')

torch.save(Network.decoder.state_dict(),
           ROOT_WEIGHTS+'/BACKUP_GOOD_LFWTRIPLETDECODER.pt')

torch.save(Network.classifier.state_dict(),
           './weights/BACKUP_GOOD_LFWTRIPLETCLASSIFIER.pt')

torch.save(Network.classifier.state_dict(), PATH_WEIGHTS_CLASSIFIER)
torch.save(Network.decoder.state_dict(), PATH_WEIGHTS_DECODER)
torch.save(Network.encoder.state_dict(), PATH_WEIGHTS_ENCODER)

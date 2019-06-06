import torch

<<<<<<< HEAD
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
NB_EPOCHS = 150
ROOT_WEIGHTS = './weights/'
ROOT_IMAGE = './images/'
ROOT_DATASET = './dataset/mp4/'
=======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NB_EPOCHS = 150
ROOT_WEIGHTS = './weights/'
ROOT_IMAGE = './images/'
>>>>>>> ac30765079493b5fb6f8b925301f46cd6ba6f86a
PATH_WEIGHTS_EMBEDDER = ROOT_WEIGHTS+'Embedder.pt'
PATH_WEIGHTS_GENERATOR = ROOT_WEIGHTS+'Generator.pt'
PATH_WEIGHTS_DISCRIMINATOR = ROOT_WEIGHTS+'Discriminator.pt'
IMAGE_SIZE = 244
<<<<<<< HEAD
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
LATENT_SIZE = 512
=======
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
>>>>>>> ac30765079493b5fb6f8b925301f46cd6ba6f86a

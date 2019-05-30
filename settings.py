import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NB_EPOCHS = 150
ROOT_WEIGHTS = './weights/'
ROOT_IMAGE = './images/'
ROOT_DATASET = './dataset/mp4/'
PATH_WEIGHTS_EMBEDDER = ROOT_WEIGHTS+'Embedder.pt'
PATH_WEIGHTS_GENERATOR = ROOT_WEIGHTS+'Generator.pt'
PATH_WEIGHTS_DISCRIMINATOR = ROOT_WEIGHTS+'Discriminator.pt'
IMAGE_SIZE = 244
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NB_EPOCHS = 40
MODEL = "small"

ROOT_WEIGHTS = './weights_beluga/'
ROOT_IMAGE = './images/'
ROOT_DATASET = './dataset/mp4/'

PATH_WEIGHTS_EMBEDDER = ROOT_WEIGHTS+'Embedder.pt'
PATH_WEIGHTS_GENERATOR = ROOT_WEIGHTS+'Generator.pt'
PATH_WEIGHTS_DISCRIMINATOR = ROOT_WEIGHTS + 'Discriminator.pt'

PATH_WEIGHTS_BIG_EMBEDDER = ROOT_WEIGHTS+'BigEmbedder.pt'
PATH_WEIGHTS_BIG_GENERATOR = ROOT_WEIGHTS+'BigGenerator.pt'
PATH_WEIGHTS_BIG_DISCRIMINATOR = ROOT_WEIGHTS + 'BigDiscriminator.pt'

LOAD_BATCH_SIZE = torch.cuda.device_count() * 1
BATCH_SIZE = LOAD_BATCH_SIZE//torch.cuda.device_count()

LEARNING_RATE_EMB = 5e-5
LEARNING_RATE_GEN = 5e-5
LEARNING_RATE_DISC = 2e-4

LATENT_SIZE = 512
K_SHOT = 8

LOAD_EMBEDDINGS = False

PRINT_EVERY = 50
LOAD_PREVIOUS = True
NB_WORKERS = 0

###############
# RL SETTINGS #
###############
GAMMA = 0.999
BATCH_SIZE_RL = 2
LEARNING_RATE_RL = 0.01
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

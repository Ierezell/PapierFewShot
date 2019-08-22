import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
<<<<<<< HEAD

NB_EPOCHS = 40
MODEL = "big"

# Weights
ROOT_WEIGHTS = './weights/'
ROOT_IMAGE = './images/'
ROOT_DATASET = './dataset/mp4/'
# ROOT_DATASET = '../scratch/dev/mp4/' # BELUGA
# ROOT_DATASET ='/scratch/syi-200-aa/dev/mp4/' # HELIOS

# Save
=======
NB_EPOCHS = 40
MODEL = "small"

ROOT_WEIGHTS = './weights_beluga/'
ROOT_IMAGE = './images/'
ROOT_DATASET = './dataset/mp4/'

>>>>>>> d23d6bbcfb8c1d6a94c0b9fc5cb92bb806ed1a86
PATH_WEIGHTS_EMBEDDER = ROOT_WEIGHTS+'Embedder.pt'
PATH_WEIGHTS_GENERATOR = ROOT_WEIGHTS+'Generator.pt'
PATH_WEIGHTS_DISCRIMINATOR = ROOT_WEIGHTS + 'Discriminator.pt'

PATH_WEIGHTS_BIG_EMBEDDER = ROOT_WEIGHTS+'BigEmbedder.pt'
PATH_WEIGHTS_BIG_GENERATOR = ROOT_WEIGHTS+'BigGenerator.pt'
PATH_WEIGHTS_BIG_DISCRIMINATOR = ROOT_WEIGHTS + 'BigDiscriminator.pt'

<<<<<<< HEAD
# Batch
nb_batch_per_gpu = 1
LOAD_BATCH_SIZE = torch.cuda.device_count() * nb_batch_per_gpu
BATCH_SIZE = LOAD_BATCH_SIZE//torch.cuda.device_count()

# LR
=======
LOAD_BATCH_SIZE = torch.cuda.device_count() * 1
BATCH_SIZE = LOAD_BATCH_SIZE//torch.cuda.device_count()

>>>>>>> d23d6bbcfb8c1d6a94c0b9fc5cb92bb806ed1a86
LEARNING_RATE_EMB = 5e-5
LEARNING_RATE_GEN = 5e-5
LEARNING_RATE_DISC = 2e-4

<<<<<<< HEAD
# Misc Size
LATENT_SIZE = 512
K_SHOT = 8

# Load parameters
LOAD_EMBEDDINGS = True
LOAD_PREVIOUS = True

DEVICE_LANDMARKS = "cpu"  # Or "cuda"
NB_WORKERS = 3


PRINT_EVERY = 100
=======
LATENT_SIZE = 512
K_SHOT = 8

LOAD_EMBEDDINGS = False

PRINT_EVERY = 50
LOAD_PREVIOUS = True
NB_WORKERS = 0
>>>>>>> d23d6bbcfb8c1d6a94c0b9fc5cb92bb806ed1a86

###############
# RL SETTINGS #
###############
GAMMA = 0.999
<<<<<<< HEAD
BATCH_SIZE_RL = 1
=======
BATCH_SIZE_RL = 2
>>>>>>> d23d6bbcfb8c1d6a94c0b9fc5cb92bb806ed1a86
LEARNING_RATE_RL = 0.01
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
<<<<<<< HEAD

PATH_WEIGHTS_POLICY = ROOT_WEIGHTS+'Policy.pt'
=======
>>>>>>> d23d6bbcfb8c1d6a94c0b9fc5cb92bb806ed1a86

import torch
import platform
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


NB_EPOCHS = 40
MODEL = "small"

# Weights
ROOT_WEIGHTS = './weights/'
ROOT_IMAGE = './images/'
if platform.system() == "Windows":
    ROOT_DATASET = '.\\dataset\\mp4'  # window
else:
    ROOT_DATASET = './dataset/mp4'  # mac & linux
    # ROOT_DATASET ='/scratch/syi-200-aa/dev/mp4/' # HELIOS
    # ROOT_DATASET = '../scratch/dev/mp4/' # BELUGA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save

PATH_WEIGHTS_EMBEDDER = ROOT_WEIGHTS+'Embedder.pt'
PATH_WEIGHTS_GENERATOR = ROOT_WEIGHTS+'Generator.pt'
PATH_WEIGHTS_DISCRIMINATOR = ROOT_WEIGHTS + 'Discriminator.pt'

PATH_WEIGHTS_BIG_EMBEDDER = ROOT_WEIGHTS+'BigEmbedder.pt'
PATH_WEIGHTS_BIG_GENERATOR = ROOT_WEIGHTS+'BigGenerator.pt'
PATH_WEIGHTS_BIG_DISCRIMINATOR = ROOT_WEIGHTS + 'BigDiscriminator.pt'


# Batch
nb_batch_per_gpu = 2
LOAD_BATCH_SIZE = torch.cuda.device_count() * nb_batch_per_gpu
BATCH_SIZE = LOAD_BATCH_SIZE//torch.cuda.device_count()

# LR

LEARNING_RATE_EMB = 5e-5
LEARNING_RATE_GEN = 5e-5
LEARNING_RATE_DISC = 2e-4

# Misc Size
LATENT_SIZE = 512
K_SHOT = 8

# Load parameters
LOAD_EMBEDDINGS = True
LOAD_PREVIOUS = True
LOAD_PREVIOUS_RL = False

DEVICE_LANDMARKS = "cuda"  # cuda or cpu
NB_WORKERS = 0


PRINT_EVERY = 100

###############
# RL SETTINGS #
###############
GAMMA = 0.999
LEARNING_RATE_RL = 0.01
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
PATH_WEIGHTS_POLICY = ROOT_WEIGHTS+'Policy.pt'

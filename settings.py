import platform

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NB_EPOCHS = 40
MODEL = "small"

# Weights
ROOT_WEIGHTS = './weights/'
ROOT_IMAGE = './images/'

if platform.system() == "Windows":
    ROOT_DATASET = '.\\dataset\\mp4'
else:
    if "blg" in platform.node():
        ROOT_DATASET = '../scratch/dev/mp4/'
    elif "gpu-k" in platform.node():
        ROOT_DATASET = '/scratch/syi-200-aa/dev/mp4/'
    else:
        ROOT_DATASET = './dataset/mp4'

# Save
# TODO transforner pour wandb
PATH_WEIGHTS_EMBEDDER = ROOT_WEIGHTS+'Embedder.pt'
PATH_WEIGHTS_GENERATOR = ROOT_WEIGHTS+'Generator.pt'
PATH_WEIGHTS_DISCRIMINATOR = ROOT_WEIGHTS + 'Discriminator.pt'

PATH_WEIGHTS_BIG_EMBEDDER = ROOT_WEIGHTS+'BigEmbedder.pt'
PATH_WEIGHTS_BIG_GENERATOR = ROOT_WEIGHTS+'BigGenerator.pt'
PATH_WEIGHTS_BIG_DISCRIMINATOR = ROOT_WEIGHTS + 'BigDiscriminator.pt'


# Batch
if "blg" in platform.node():
    nb_batch_per_gpu = 6
elif "gpu-k" in platform.node():
    nb_batch_per_gpu = 4
elif "GATINEAU" in platform.node():
    nb_batch_per_gpu = 2
else:
    nb_batch_per_gpu = 1


LOAD_BATCH_SIZE = torch.cuda.device_count() * nb_batch_per_gpu
# BATCH_SIZE = LOAD_BATCH_SIZE//torch.cuda.device_count()
BATCH_SIZE = nb_batch_per_gpu

# LR
LEARNING_RATE_EMB = 5e-6
LEARNING_RATE_GEN = 5e-6
LEARNING_RATE_DISC = 1e-5

# Sizes
LATENT_SIZE = 512
K_SHOT = 8

# Load parameters
LOAD_EMBEDDINGS = False
LOAD_PREVIOUS = False
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
MAX_DEQUE_LANDMARKS = 1000
MAX_ITER_PERSON = 50


CONFIG = {
    "platform": platform.node(),
    "batch_size": BATCH_SIZE,
    "lr_gen": LEARNING_RATE_GEN,
    "lr_disc": LEARNING_RATE_DISC,
    "resume": LOAD_PREVIOUS,
    "nb_gpu": torch.cuda.device_count(),
    "k_shot": K_SHOT,
    "model": MODEL,
    "disc_out": "div 10",
    "in_disc": "noisy"
}


CONFIG_RL = {"batch_size": BATCH_SIZE,
             "lr": LEARNING_RATE_RL,
             "decay_start": EPS_START,
             "decay_end": EPS_END,
             "decay": EPS_DECAY,
             "max_iter_person": MAX_ITER_PERSON,
             "max_deque": MAX_DEQUE_LANDMARKS,
             }

# find /some/directory -maxdepth 1 -type f -exec cmd option {} \; > results.out

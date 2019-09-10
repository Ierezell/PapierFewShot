import os
import platform
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NB_EPOCHS = 40
MODEL = "small"
LAYERS = "big"
CONCAT = True

# Weights
ROOT_WEIGHTS = './weights/'

if platform.system() == "Windows":
    ROOT_DATASET = '.\\dataset\\mp4'
else:
    if "blg" in platform.node():
        ROOT_DATASET = '../scratch/dev/mp4/'
    elif "gpu-k" in platform.node():
        ROOT_DATASET = '/scratch/syi-200-aa/dev/mp4/'
    else:
        ROOT_DATASET = './dataset/mp4'

# Batch
if "blg" in platform.node():
    nb_batch_per_gpu = 6
elif "gpu-k" in platform.node():
    nb_batch_per_gpu = 4
else:
    nb_batch_per_gpu = 1


LOAD_BATCH_SIZE = torch.cuda.device_count() * nb_batch_per_gpu
BATCH_SIZE = LOAD_BATCH_SIZE//torch.cuda.device_count()

# LR
LEARNING_RATE_EMB = 5e-6
LEARNING_RATE_GEN = 5e-6
LEARNING_RATE_DISC = 5e-6

# Sizes
LATENT_SIZE = 512
K_SHOT = 8


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
MAX_DEQUE_LANDMARKS = 1000
MAX_ITER_PERSON = 50


CONFIG = {
    "platform": platform.node()[:4],
    "batch_size": BATCH_SIZE,
    "lr_gen": LEARNING_RATE_GEN,
    "lr_disc": LEARNING_RATE_DISC,
    "nb_gpu": torch.cuda.device_count(),
    "k_shot": K_SHOT,
    "model": MODEL,
    "layers": LAYERS,
    "disc_out": "div10",
    "in_disc": "noisy",
    "concat": True,
}


CONFIG_RL = {"batch_size": BATCH_SIZE,
             "lr": LEARNING_RATE_RL,
             "decay_start": EPS_START,
             "decay_end": EPS_END,
             "decay": EPS_DECAY,
             "max_iter_person": MAX_ITER_PERSON,
             "max_deque": MAX_DEQUE_LANDMARKS,
             }

folder_weights = str(CONFIG['model'])+'_'+str(CONFIG['batch_size'])+'_' +\
    str(CONFIG['disc_out'])+'_'+str(CONFIG['in_disc'])+'_' +\
    str(CONFIG['k_shot'])+'_'+str(CONFIG['layers']) +\
    str(CONFIG['lr_gen'])+'_'+str(CONFIG['lr_disc'])+'/'

# Load parameters
if not os.path.exists(ROOT_WEIGHTS+folder_weights):
    os.makedirs(ROOT_WEIGHTS + folder_weights)
    LOAD_EMBEDDINGS = False
    LOAD_PREVIOUS = False
    LOAD_PREVIOUS_RL = False

else:
    LOAD_EMBEDDINGS = True
    LOAD_PREVIOUS = True
    LOAD_PREVIOUS_RL = True

CONFIG["resume"] = LOAD_PREVIOUS

# Save
PATH_WEIGHTS_EMBEDDER = ROOT_WEIGHTS+folder_weights+'Embedder.pt'
PATH_WEIGHTS_GENERATOR = ROOT_WEIGHTS+folder_weights+'Generator.pt'
PATH_WEIGHTS_DISCRIMINATOR = ROOT_WEIGHTS + folder_weights + 'Discriminator.pt'

PATH_WEIGHTS_POLICY = ROOT_WEIGHTS+'Policy.pt'

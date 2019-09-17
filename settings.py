import os
import platform
import torch
from pathlib import Path
import datetime


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NB_EPOCHS = 40
MODEL = "small"
LAYERS = "big"
CONCAT = True

ROOT_WEIGHTS = './weights/'

# Weights
if "blg" in platform.node():
    ROOT_DATASET = '../scratch/dev/mp4/'
elif "gpu-k" in platform.node():
    ROOT_DATASET = '/scratch/syi-200-aa/dev/mp4/'
else:
    ROOT_DATASET = Path('./dataset/mp4')

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
LEARNING_RATE_DISC = 5e-5
TTUR = True

# Sizes
LATENT_SIZE = 512
K_SHOT = 16


DEVICE_LANDMARKS = "cpu"  # cuda or cpu
NB_WORKERS = 8

PRINT_EVERY = 1000

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

TIME = str(datetime.datetime.now().replace(microsecond=0)
           ).replace(" ", "_").replace(":", "-")

CONFIG = {
    "PLATFORM": str(platform.node()[:4]),
    "BATCH_SIZE": str(BATCH_SIZE),
    "LR_GEN": str(LEARNING_RATE_GEN),
    "LR_DISC": str(LEARNING_RATE_DISC),
    "NB_GPU": str(torch.cuda.device_count()),
    "K_SHOT": str(K_SHOT),
    "MODEL": str(MODEL),
    "LAYERS": str(LAYERS),
    "DISC_OUT": "tanh",
    "IN_DISC": "noisy",
    "CONCAT": str(CONCAT),
    "TTUR": str(TTUR),
    "TIME": TIME,
}
folder_weights = CONFIG["PLATFORM"] + "_" + CONFIG["BATCH_SIZE"] + "_" + \
    CONFIG["LR_GEN"]+"_" + CONFIG["LR_DISC"]+"_" +\
    CONFIG["NB_GPU"] + "_" + CONFIG["K_SHOT"] + "_" + \
    CONFIG["MODEL"] + "_" + CONFIG["LAYERS"] + "_" + \
    CONFIG["DISC_OUT"] + "_" + CONFIG["IN_DISC"] + "_" + \
    CONFIG["CONCAT"]+"_" + CONFIG["TTUR"]+"/"

CONFIG_RL = {"batch_size": str(BATCH_SIZE),
             "lr": str(LEARNING_RATE_RL),
             "decay_start": str(EPS_START),
             "decay_end": str(EPS_END),
             "decay": str(EPS_DECAY),
             "max_iter_person": str(MAX_ITER_PERSON),
             "max_deque": str(MAX_DEQUE_LANDMARKS),
             }

folder_weights_Rl = CONFIG_RL['batch_size']+'_'+CONFIG_RL['lr']+'_' +\
    CONFIG_RL['decay_start']+'_'+CONFIG_RL['decay_end']+'_' +\
    CONFIG_RL['decay']+'_'+CONFIG_RL['max_iter_person'] + '_' +\
    CONFIG_RL['max_deque']+'/'

# ##########
# Override #
# ##########
# folder_weights = "/Beluga/"

# Load parameters
if not os.path.exists(ROOT_WEIGHTS+folder_weights):
    os.makedirs(ROOT_WEIGHTS + folder_weights)
    LOAD_EMBEDDINGS = False
    LOAD_PREVIOUS = False
else:
    LOAD_EMBEDDINGS = True
    LOAD_PREVIOUS = True

if not os.path.exists(ROOT_WEIGHTS+folder_weights_Rl):
    os.makedirs(ROOT_WEIGHTS + folder_weights_Rl)
    LOAD_PREVIOUS_RL = False
else:
    LOAD_PREVIOUS_RL = True

# Save
PATH_WEIGHTS_EMBEDDER = ROOT_WEIGHTS+folder_weights+'Embedder.pt'
PATH_WEIGHTS_GENERATOR = ROOT_WEIGHTS+folder_weights+'Generator.pt'
PATH_WEIGHTS_DISCRIMINATOR = ROOT_WEIGHTS + folder_weights + 'Discriminator.pt'
PATH_WEIGHTS_POLICY = ROOT_WEIGHTS+folder_weights_Rl+'Policy.pt'

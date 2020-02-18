import datetime
import os
import platform

import torch
import wandb

PLATFORM = platform.node()[:3]

if ("blg" in PLATFORM) or ("gpu" in PLATFORM):  # or ("Arc" in PLATFORM):
    os.environ['WANDB_MODE'] = 'dryrun'

wandb.init(project="papier_few_shot", entity="plop", reinit=True)
wandb.run.config['PLATFORM'] = PLATFORM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARALLEL = False
if torch.cuda.device_count() > 1:
    PARALLEL = True

LEARNING_RATE_DISC = wandb.config.LEARNING_RATE_DISC
LEARNING_RATE_EMB = wandb.config.LEARNING_RATE_EMB
LEARNING_RATE_GEN = wandb.config.LEARNING_RATE_GEN
ROOT_WEIGHTS = wandb.config.ROOT_WEIGHTS
LATENT_SIZE = wandb.config.LATENT_SIZE
NB_WORKERS = wandb.config.NB_WORKERS
NB_EPOCHS = wandb.config.NB_EPOCHS
ATTENTION = wandb.config.ATTENTION
DATASET = wandb.config.DATASET
IN_DISC = wandb.config.IN_DISC
LAYERS = wandb.config.LAYERS
CONCAT = wandb.config.CONCAT
LOADER = wandb.config.LOADER
K_SHOT = wandb.config.K_SHOT
MODEL = wandb.config.MODEL
TTUR = wandb.config.TTUR
HALF = wandb.config.HALF
TTUR = False
if platform.system() == "Windows":
    NB_WORKERS = 0

if LOADER == "json":
    if DATASET == "big":
        if "blg" in PLATFORM:
            ROOT_DATASET = '../scratch/dev/jsonDataset'
        elif "gpu" in PLATFORM:
            ROOT_DATASET = '/scratch/syi-200-aa/dev/jsonDataset'
        elif "GAT" in PLATFORM:
            ROOT_DATASET = "H:\\dataset\\voxCeleb\\dev\\jsonDataset"
        elif "co" in PLATFORM:
            ROOT_DATASET = '/home-local2/pisne.extra.nobkp/dataset/dev/jsonDataset'
        else:
            ROOT_DATASET = "/run/media/pedro/Elements/dataset/voxCeleb/dev/jsonDataset"
    elif DATASET == "small":
        ROOT_DATASET = './dataset/jsonDataset'
    ROOT_FINE_TUNING_DATASET = './dataset/fine_tuning'

IMAGE_SIZE = (224, 224)
# Batch
if "blg" in PLATFORM:
    if MODEL == "small":
        BATCH_SIZE = 8
    elif MODEL == "big":
        BATCH_SIZE = 4
    BATCH_SIZE_LDMK = 64
elif ("gpu" in PLATFORM) or ("GAT" in PLATFORM) or ("coi" in PLATFORM):
    if MODEL == "small":
        BATCH_SIZE = 4
    elif MODEL == "big":
        BATCH_SIZE = 4
    BATCH_SIZE_LDMK = 32
elif "Arc" in PLATFORM:
    ROOT_DATASET = './dataset/jsonDataset'
    BATCH_SIZE = 1
    K_SHOT = 8
    LATENT_SIZE = 512
    NB_WORKERS = 12
    MODEL = "small"
    BATCH_SIZE_LDMK = 1


LOAD_BATCH_SIZE = BATCH_SIZE * (torch.cuda.device_count()
                                if torch.cuda.is_available()
                                else 1)

LOAD_BATCH_SIZE_LDMK = BATCH_SIZE_LDMK * (torch.cuda.device_count()
                                          if torch.cuda.is_available()
                                          else 1)


###############
# RL SETTINGS #
###############
GAMMA = wandb.config.GAMMA
LEARNING_RATE_RL = wandb.config.LEARNING_RATE_RL
EPS_START = wandb.config.EPS_START
EPS_END = wandb.config.EPS_END
EPS_DECAY = wandb.config.EPS_DECAY
MAX_DEQUE_LANDMARKS = wandb.config.MAX_DEQUE_LANDMARKS
MAX_ITER_PERSON = wandb.config.MAX_ITER_PERSON
TRAIN_RL = wandb.config.TRAIN_RL

if TRAIN_RL:
    BATCH_SIZE = 1

TIME = str(datetime.datetime.now().replace(microsecond=0)
           ).replace(" ", "_").replace(":", "-")

CONFIG = {
    "PLATFORM": str(platform.node()[:3]),
    "BATCH_SIZE": str(BATCH_SIZE),
    "DATASET": str(DATASET),
    "LR_GEN": str(LEARNING_RATE_GEN),
    "LR_DISC": str(LEARNING_RATE_DISC),
    "NB_GPU": str(torch.cuda.device_count()),
    "K_SHOT": str(K_SHOT),
    "LATENT_SIZE": str(LATENT_SIZE),
    "MODEL": str(MODEL),
    "LAYERS": str(LAYERS),
    "IN_DISC": IN_DISC,
    "CONCAT": str(CONCAT),
    "TTUR": str(TTUR),
    "TIME": TIME,
}

folder_weights = CONFIG["PLATFORM"] + "_"+CONFIG["DATASET"] + "_" + \
    CONFIG["BATCH_SIZE"] + "_" + \
    CONFIG["LR_GEN"]+"_" + CONFIG["LR_DISC"]+"_" +\
    CONFIG["K_SHOT"] + "_" + \
    CONFIG["MODEL"] + "_" + CONFIG["LAYERS"] + "_" + \
    CONFIG["IN_DISC"] + "_" + \
    CONFIG["CONCAT"]+"_" + CONFIG["TTUR"] + "_" + CONFIG["LATENT_SIZE"] + "/"

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

# Load parameters
if not os.path.exists(ROOT_WEIGHTS+folder_weights):
    os.makedirs(ROOT_WEIGHTS + folder_weights)
    LOAD_PREVIOUS = False
else:
    LOAD_PREVIOUS = True

if not os.path.exists(ROOT_WEIGHTS+folder_weights_Rl):
    os.makedirs(ROOT_WEIGHTS + folder_weights_Rl)
    LOAD_PREVIOUS_RL = False
else:
    LOAD_PREVIOUS_RL = False

# Save
# PATH_WEIGHTS_ROOT = ROOT_WEIGHTS+folder_weights
PATH_WEIGHTS_ROOT = ROOT_WEIGHTS+"/blg_small_8_5e-05_5e-05_8_small_big___False_512/"
PATH_WEIGHTS_EMBEDDER = PATH_WEIGHTS_ROOT+'Embedder.pt'
PATH_WEIGHTS_GENERATOR = PATH_WEIGHTS_ROOT+'Generator.pt'
PATH_WEIGHTS_DISCRIMINATOR = PATH_WEIGHTS_ROOT + 'Discriminator.pt'
PATH_WEIGHTS_POLICY = ROOT_WEIGHTS+folder_weights_Rl+'Policy.pt'

# PATH_WEIGHTS_EMBEDDER = "./weights/top_small/"+'Embedder.pt'
# PATH_WEIGHTS_GENERATOR = "./weights/top_small/"+'Generator.pt'
# PATH_WEIGHTS_DISCRIMINATOR = "./weights/top_small/" + 'Discriminator.pt'
print(folder_weights)

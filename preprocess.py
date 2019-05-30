from face_alignment import FaceAlignment, LandmarksType
from torch.utils.data import DataLoader, Dataset
from settings import ROOT_DATASET, BATCH_SIZE
import glob
import torch
from torchvision import transforms

import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import datetime
import itertools
import glob
import random
from PIL import Image
import torch.nn as nn
import torchvision.utils as vutils
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torchvision import models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_landmarks(image):
    fa = FaceAlignment(LandmarksType._3D,
                       flip_input=False, device="cpu")  # device="cuda:0")
    return fa.get_landmarks(image)


def get_landmarks_folder(path_folder):
    fa = FaceAlignment(LandmarksType._3D,
                       flip_input=False, device="cpu")  # device="cuda:0")
    return fa.get_landmarks_from_directory(path_folder)


class frameLoader(Dataset):
    def __init__(self, root_dir=ROOT_DATASET):
        super(frameLoader, self).__init__()
        self.root_dir = root_dir
        self.ids = glob.glob(f"{self.root_dir}/*")
        self.contexts = glob.glob(f"{self.root_dir}/*/*")
        self.mp4files = glob.glob(f"{self.root_dir}/*/*/*")

        self.id_to_int = {name.split('/')[-1]: torch.tensor([i])
                          for i, name in enumerate(self.ids)}

    def __getitem__(self, index):
        mp4File = np.random.choice(self.mp4files)
        video = cv2.VideoCapture(mp4File)
        total_frame_nb = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frameIndex = np.random.randint(0, total_frame_nb)
        video.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        frame = video.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_tensor = transforms.ToTensor(img)

        contextIndex = (frameIndex+(total_frame_nb//10)) % total_frame_nb
        video.set(cv2.CAP_PROP_POS_FRAMES, contextIndex)
        context_img_tensors = []
        for i in range(8):
            contextFrame = video.read()
            context_img = cv2.cvtColor(contextFrame, cv2.COLOR_BGR2RGB)
            context_img_tensors.append(transforms.ToTensor(context_img))

        return im_tensor, context_img_tensors

    def __len__(self):
        return len(self.mp4files)


def get_data_loader():
    datas = frameLoader()
    # print(len(datas))
    # size_train = int(0.8 * len(datas))
    # size_valid = len(datas) - int(0.8 * len(datas))
    # train_datas, valid_datas = random_split(datas, (size_train, size_valid))
    train_loader = DataLoader(datas, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)
    return train_loader


def view_batch():
    datas = frameLoader()
    # print(len(datas))
    # size_train = int(0.8 * len(datas))
    # size_valid = len(datas) - int(0.8 * len(datas))
    # train_datas, valid_datas = random_split(datas, (size_train, size_valid))
    train_loader = DataLoader(datas, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)
    real_batch = next(train_loader)
    batch_view = torch.stack(real_batch[0], dim=1).view(-1, 3, 224, 224)
    # %matplotlib inline
    plt.figure(figsize=(25, 10))
    plt.axis("off")
    plt.title("Training Images exemple\n\nAnchor    Positive  Negative")
    plt.imshow(np.transpose(vutils.make_grid(batch_view, padding=2,
                                             nrow=6, normalize=True).cpu(),
                            (1, 2, 0)))
    plt.show()

from face_alignment import FaceAlignment, LandmarksType
from torch.utils.data import DataLoader, Dataset
from settings import ROOT_DATASET, BATCH_SIZE, DEVICE, K_SHOT
import glob
import torch

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
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torchvision import models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR


class frameLoader(Dataset):
    def __init__(self, root_dir=ROOT_DATASET, K_shots=K_SHOT):
        super(frameLoader, self).__init__()
        self.face_landmarks = FaceAlignment(
            LandmarksType._2D, device="cuda")
        self.K_shots = K_shots
        self.root_dir = root_dir
        self.ids = glob.glob(f"{self.root_dir}/*")
        self.contexts = glob.glob(f"{self.root_dir}/*/*")
        self.mp4files = glob.glob(f"{self.root_dir}/*/*/*")

        self.id_to_tensor = {name.split('/')[-1]: torch.tensor(i).view(1)
                             for i, name in enumerate(self.ids)}

    def load_random(self, video, total_frame_nb, fusion):
        frameIndex = np.random.randint(0, total_frame_nb)
        video.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        _, gt_im = video.read()
        gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB)

        landmarks = self.face_landmarks.get_landmarks_from_image(gt_im)
        landmarks = landmarks[0]
        if not fusion:
            image = np.zeros(gt_im.shape, np.float32)
        else:
            image = gt_im

        # Machoire
        cv2.polylines(image, [np.int32(landmarks[0:17])],
                      isClosed=False, color=(0, 255, 0))
        # Sourcil Gauche
        cv2.polylines(image, [np.int32(landmarks[17:22])],
                      isClosed=False, color=(255, 0, 0))
        # Sourcil droit
        cv2.polylines(image, [np.int32(landmarks[22:27])],
                      isClosed=False, color=(255, 0, 0))
        # Nez arrete
        cv2.polylines(image, [np.int32(landmarks[27:31])],
                      isClosed=False, color=(255, 0, 255))
        # Nez narine
        cv2.polylines(image, [np.int32(landmarks[31:36])],
                      isClosed=False, color=(255, 0, 255))
        # Oeil gauche
        cv2.polylines(image, [np.int32(landmarks[36:42])],
                      isClosed=True, color=(0, 0, 255))
        # oeil droit
        cv2.polylines(image, [np.int32(landmarks[42:48])],
                      isClosed=True, color=(0, 0, 255))
        # Bouche exterieur
        cv2.polylines(image, [np.int32(landmarks[48:60])],
                      isClosed=True, color=(255, 255, 0))
        # Bouche interieur
        cv2.polylines(image, [np.int32(landmarks[60:68])],
                      isClosed=True, color=(255, 255, 0))

        if not fusion:
            return transforms.ToTensor()(gt_im), transforms.ToTensor()(image)
        else:
            return transforms.ToTensor()(image)

    def __getitem__(self, index):
        mp4File = self.mp4files[index]
        itemId = self.id_to_tensor[mp4File.split('/')[-3]]
        video = cv2.VideoCapture(mp4File)
        total_frame_nb = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        gt_im_tensor, gt_landmarks = self.load_random(video,
                                                      total_frame_nb,
                                                      fusion=False)

        context_tensors_list = []
        for _ in range(self.K_shots):
            context_frame = self.load_random(video, total_frame_nb,
                                             fusion=True)
            context_tensors_list.append(context_frame)

        context_tensors = torch.cat(context_tensors_list)
        video.release()
        return gt_im_tensor, gt_landmarks, context_tensors, itemId

    def __len__(self):
        return len(self.mp4files)


def get_data_loader(root_dir=ROOT_DATASET, K_shots=8, workers=0):
    datas = frameLoader(root_dir=root_dir, K_shots=K_shots)
    # print(len(datas))
    # size_train = int(0.8 * len(datas))
    # size_valid = len(datas) - int(0.8 * len(datas))
    # train_datas, valid_datas = random_split(datas, (size_train, size_valid))
    pin = False if DEVICE.type == 'cpu' else True
    train_loader = DataLoader(datas, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=workers, pin_memory=pin)
    return train_loader, len(datas.ids)


def view_batch(loader):
    gt_im, gt_landmarks, context, itemId = next(loader)
    grid = torchvision.utils.make_grid(
        torch.cat((gt_im, gt_landmarks, context), dim=1).view(-1, 3, 224, 224),
        nrow=2 + K_SHOT, padding=2, normalize=True).cpu()
    # %matplotlib inline
    plt.figure(figsize=(25, 10))
    plt.axis("off")
    plt.title("Training Images exemple\n\nAnchor    Positive  Negative")
    plt.imshow(np.transpose(grid))
    plt.show()

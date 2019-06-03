from face_alignment import FaceAlignment, LandmarksType
from torch.utils.data import DataLoader, Dataset
from settings import ROOT_DATASET, BATCH_SIZE, DEVICE
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
    def __init__(self, root_dir=ROOT_DATASET, K_shots=8):
        super(frameLoader, self).__init__()
        self.face_landmarks = FaceAlignment(LandmarksType._2D,
                                            flip_input=False,
                                            device=str(DEVICE))
        # device="cuda:0")
        self.K_shots = K_shots
        self.root_dir = root_dir
        self.ids = glob.glob(f"{self.root_dir}/*")
        self.contexts = glob.glob(f"{self.root_dir}/*/*")
        self.mp4files = glob.glob(f"{self.root_dir}/*/*/*")

        self.id_to_int = {name.split('/')[-1]: torch.tensor([i])
                          for i, name in enumerate(self.ids)}

    def __getitem__(self, index):
        # mp4File = np.random.choice(self.mp4files)
        mp4File = self.mp4files[index]
        # print(mp4File)
        video = cv2.VideoCapture(mp4File)
        total_frame_nb = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frameIndex = np.random.randint(0, total_frame_nb)
        video.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        gt_im = video.read()[1]
        landmark_image = np.zeros(gt_im.shape, np.int32)
        landmarks = self.face_landmarks.get_landmarks(gt_im)[0]
        # Machoire
        cv2.polylines(landmark_image, [np.int32(landmarks[0:17])],
                      isClosed=False, color=(0, 255, 0))
        # Sourcil Gauche
        cv2.polylines(landmark_image, [np.int32(landmarks[17:22])],
                      isClosed=False, color=(255, 0, 0))
        # Sourcil droit
        cv2.polylines(landmark_image, [np.int32(landmarks[22:27])],
                      isClosed=False, color=(255, 0, 0))
        # Nez arrete
        cv2.polylines(landmark_image, [np.int32(landmarks[27:31])],
                      isClosed=False, color=(255, 0, 255))
        # Nez narine
        cv2.polylines(landmark_image, [np.int32(landmarks[31:36])],
                      isClosed=False, color=(255, 0, 255))
        # Oeil gauche
        cv2.polylines(landmark_image, [np.int32(landmarks[36:42])],
                      isClosed=True, color=(0, 0, 255))
        # oeil droit
        cv2.polylines(landmark_image, [np.int32(landmarks[42:48])],
                      isClosed=True, color=(0, 0, 255))
        # Bouche exterieur
        cv2.polylines(landmark_image, [np.int32(landmarks[48:60])],
                      isClosed=True, color=(255, 255, 0))
        # Bouche interieur
        cv2.polylines(landmark_image, [np.int32(landmarks[60:68])],
                      isClosed=True, color=(255, 255, 0))

        gt_im_landmarks_tensor = transforms.ToTensor()(landmark_image)
        gt_im_tensor = transforms.ToTensor()(gt_im)
        # print("gt_done")
        indexOk = False
        contextIndex = (frameIndex+(total_frame_nb//10)) % total_frame_nb
        while not indexOk:
            if contextIndex + self.K_shots > total_frame_nb:
                contextIndex -= 1
                print(f"reduce ! frameIndex is {frameIndex} :",
                      f"contextIndex was {contextIndex} on {total_frame_nb}")
            else:
                indexOk = True

        video.set(cv2.CAP_PROP_POS_FRAMES, contextIndex)
        context_tensors = []
        for i in range(self.K_shots):
            contextFrame = video.read()[1]
            ctxt_ldmk_img = np.zeros(contextFrame.shape, np.float32)
            ctxt_landmarks = self.face_landmarks.get_landmarks(contextFrame)[0]
            context_tensors.append(transforms.ToTensor()(contextFrame))

            # Machoire
            cv2.polylines(ctxt_ldmk_img, [np.int32(ctxt_landmarks[0:17])],
                          isClosed=False, color=(0, 255, 0))
            # Sourcil Gauche
            cv2.polylines(ctxt_ldmk_img, [np.int32(ctxt_landmarks[17:22])],
                          isClosed=False, color=(255, 0, 0))
            # Sourcil droit
            cv2.polylines(ctxt_ldmk_img, [np.int32(ctxt_landmarks[22:27])],
                          isClosed=False, color=(255, 0, 0))
            # Nez arrete
            cv2.polylines(ctxt_ldmk_img, [np.int32(ctxt_landmarks[27:31])],
                          isClosed=False, color=(255, 0, 255))
            # Nez narine
            cv2.polylines(ctxt_ldmk_img, [np.int32(ctxt_landmarks[31:36])],
                          isClosed=False, color=(255, 0, 255))
            # Oeil gauche
            cv2.polylines(ctxt_ldmk_img, [np.int32(ctxt_landmarks[36:42])],
                          isClosed=True, color=(0, 0, 255))
            # oeil droit
            cv2.polylines(ctxt_ldmk_img, [np.int32(ctxt_landmarks[42:48])],
                          isClosed=True, color=(0, 0, 255))
            # Bouche exterieur
            cv2.polylines(ctxt_ldmk_img, [np.int32(ctxt_landmarks[48:60])],
                          isClosed=True, color=(255, 255, 0))
            # Bouche interieur
            cv2.polylines(ctxt_ldmk_img, [np.int32(ctxt_landmarks[60:68])],
                          isClosed=True, color=(255, 255, 0))
            context_tensors.append(transforms.ToTensor()(ctxt_ldmk_img))
        # print("context done")
        context_tensors = torch.cat(context_tensors)
        return gt_im_tensor, gt_im_landmarks_tensor, context_tensors

    def __len__(self):
        return len(self.mp4files)


def get_data_loader(root_dir=ROOT_DATASET, K_shots=8):
    datas = frameLoader(root_dir=ROOT_DATASET, K_shots=8)
    # print(len(datas))
    # size_train = int(0.8 * len(datas))
    # size_valid = len(datas) - int(0.8 * len(datas))
    # train_datas, valid_datas = random_split(datas, (size_train, size_valid))
    train_loader = DataLoader(datas, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=8)
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

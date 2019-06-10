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
import torchvision.utils as vutils
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torchvision import models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

face_landmarks = FaceAlignment(LandmarksType._2D, device='cpu')


def get_landmarks(image):
    fa = FaceAlignment(LandmarksType._3D,
                       flip_input=False, device="cpu")  # device="cuda:0")
    return fa.get_landmarks(image)


def get_landmarks_folder(path_folder):
    fa = FaceAlignment(LandmarksType._3D,
                       flip_input=False, device="cpu")  # device="cuda:0")
    return fa.get_landmarks_from_directory(path_folder)


class frameLoader(Dataset):
    def __init__(self, root_dir=ROOT_DATASET, K_shots=K_SHOT):
        super(frameLoader, self).__init__()
        # self.face_landmarks = FaceAlignment(LandmarksType._2D, device='cpu')
        self.K_shots = K_shots
        self.root_dir = root_dir
        self.ids = glob.glob(f"{self.root_dir}/*")
        self.contexts = glob.glob(f"{self.root_dir}/*/*")
        self.mp4files = glob.glob(f"{self.root_dir}/*/*/*")

        self.id_to_tensor = {name.split('/')[-1]: torch.tensor(i).view(1)
                             for i, name in enumerate(self.ids)}

    def __getitem__(self, index):
        # torch.set_grad_enabled(False)
        mp4File = self.mp4files[index]
        itemId = self.id_to_tensor[mp4File.split('/')[-3]]
        video = cv2.VideoCapture(mp4File)
        total_frame_nb = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frameIndex = np.random.randint(0, total_frame_nb)
        video.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        _, gt_im = video.read()
        gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB)

        landmarks = face_landmarks.get_landmarks(gt_im)
        landmarks = landmarks[0]
        # landmarks = np.random.rand(68, 2)
        landmark_image = np.zeros(gt_im.shape, np.float32)
        #  gt_im=np.array(gt_im, np.float32).transpose(2, 1, 0)
        gt_im_tensor = transforms.ToTensor()(gt_im)
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

        #  gt_landmarks = torch.tensor(landmark_image.transpose(2, 1, 0))
        gt_landmarks = transforms.ToTensor()(landmark_image)
        indexOk = False
        contextIndex = (frameIndex+(total_frame_nb//10)) % total_frame_nb
        while not indexOk:
            if contextIndex + self.K_shots > total_frame_nb:
                contextIndex -= 1
            else:
                indexOk = True

        video.set(cv2.CAP_PROP_POS_FRAMES, contextIndex)
        context_tensors_list = []
        for _ in range(self.K_shots):
            _, contextFrame = video.read()
            contextFrame = cv2.cvtColor(contextFrame, cv2.COLOR_BGR2RGB)
            # ctxt_landmarks = self.face_landmarks.get_landmarks(contextFrame)[0]
            ctxt_landmarks = np.random.rand(68, 2)
            # Machoire
            cv2.polylines(contextFrame, [np.int32(ctxt_landmarks[0:17])],
                          isClosed=False, color=(0, 255, 0))
            # Sourcil Gauche
            cv2.polylines(contextFrame, [np.int32(ctxt_landmarks[17:22])],
                          isClosed=False, color=(255, 0, 0))
            # Sourcil droit
            cv2.polylines(contextFrame, [np.int32(ctxt_landmarks[22:27])],
                          isClosed=False, color=(255, 0, 0))
            # Nez arrete
            cv2.polylines(contextFrame, [np.int32(ctxt_landmarks[27:31])],
                          isClosed=False, color=(255, 0, 255))
            # Nez narine
            cv2.polylines(contextFrame, [np.int32(ctxt_landmarks[31:36])],
                          isClosed=False, color=(255, 0, 255))
            # Oeil gauche
            cv2.polylines(contextFrame, [np.int32(ctxt_landmarks[36:42])],
                          isClosed=True, color=(0, 0, 255))
            # oeil droit
            cv2.polylines(contextFrame, [np.int32(ctxt_landmarks[42:48])],
                          isClosed=True, color=(0, 0, 255))
            # Bouche exterieur
            cv2.polylines(contextFrame, [np.int32(ctxt_landmarks[48:60])],
                          isClosed=True, color=(255, 255, 0))
            # Bouche interieur
            cv2.polylines(contextFrame, [np.int32(ctxt_landmarks[60:68])],
                          isClosed=True, color=(255, 255, 0))
            # contextFrame=np.array(contextFrame,np.float32).transpose(2, 1, 0)
            # context_tensors_list.append(torch.tensor(contextFrame))
            context_tensors_list.append(transforms.ToTensor()(contextFrame))

            # context_tensors=torch.rand(self.K_shots, 224, 224)
            # torch.cat(context_tensors_list, out=context_tensors)
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
    # pin = False if DEVICE.type == 'cpu' else True
    train_loader = DataLoader(datas, batch_size=BATCH_SIZE, shuffle=True)
    # num_workers=workers, pin_memory=pin)
    return train_loader, len(datas.ids)


def view_batch(loader):
    real_batch = next(loader)
    batch_view = torch.stack(real_batch[0], dim=1).view(-1, 3, 224, 224)
    # %matplotlib inline
    plt.figure(figsize=(25, 10))
    plt.axis("off")
    plt.title("Training Images exemple\n\nAnchor    Positive  Negative")
    plt.imshow(np.transpose(vutils.make_grid(batch_view, padding=2,
                                             nrow=6, normalize=True).cpu(),
                            (1, 2, 0)))
    plt.show()

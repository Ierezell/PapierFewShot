import copy
import glob
import json
import os
import platform
import time
from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from face_alignment import FaceAlignment, LandmarksType
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from settings import (DEVICE, HALF, IMAGE_SIZE, K_SHOT, LOAD_BATCH_SIZE,
                      LOAD_BATCH_SIZE_LDMK, LOADER, NB_WORKERS, ROOT_DATASET,
                      ROOT_WEIGHTS)


def write_landmarks_on_image(image, landmarks):
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
    return image.astype(np.uint8)

# #############
# JSON LOADER #
# #############


def dictKeytoInt(x): return {int(k): v for k, v in x}


def get_ids(root_dir=ROOT_DATASET):
    if platform.system() == "Windows":
        slash = "\\"
    else:
        slash = "/"
    with open(f"{ROOT_WEIGHTS}ids.json", "w+") as file:
        try:
            json_ids = json.load(file)
        except json.decoder.JSONDecodeError:
            json_ids = {}

    current_id = -1
    id_to_tensor = {}
    for uid in glob.glob(f"{root_dir}/*"):
        key = uid.split(slash)[-1]
        id_to_tensor[key] = json_ids.get(key, current_id + 1)
        current_id = id_to_tensor[key]

    with open(f"{ROOT_WEIGHTS}/ids.json", "w") as file:
        json.dump(id_to_tensor, file)

    id_to_tensor = {key: torch.tensor(value).view(1)
                    for key, value in id_to_tensor.items()}
    return id_to_tensor


class ldmkLoader(Dataset):

    def __init__(self, root_dir=ROOT_DATASET, K_shots=K_SHOT):
        super(ldmkLoader, self).__init__()
        self.slash = "/"
        if "Windows" in platform.system():
            self.slash = "\\"
        self.K_shots = K_shots
        self.root_dir = root_dir
        # print("\tLoading ids...")
        # start_time = time.time()
        # self.ids = glob.glob(f"{self.root_dir}/*")
        # self.id_to_tensor = get_ids()
        # print(f"\tIds loaded in {time.time() - start_time}s")
        print("\tLoading videos...")
        start_time = time.time()
        self.context_names = [video[:-5] for video in
                              glob.glob(f"{self.root_dir}/*/*.json")]
        print(f"\tVideos loaded in {time.time() - start_time}s")

    def __getitem__(self, index):
        badLdmks = True
        while badLdmks:
            context_name = self.context_names[index]
            # itemId = self.id_to_tensor[context_name.split(self.slash)[-2]]
            with open(f"{context_name}.json", "r") as file:
                dict_ldmk = json.load(file, object_pairs_hook=dictKeytoInt)

            try:
                frames = np.random.choice(list(dict_ldmk.keys()), 1)
                badLdmks = False
            except ValueError:
                index = randint(0, len(self.context_names))

        gt_ldmk = dict_ldmk[frames[0]]
        gt_ldmk_im = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), np.float32)
        gt_ldmk_im = write_landmarks_on_image(gt_ldmk_im, gt_ldmk)
        gt_ldmk_im = transforms.ToPILImage()(gt_ldmk_im)
        gt_ldmk_im = transforms.Resize(IMAGE_SIZE)(gt_ldmk_im)
        gt_ldmk_im_tensor = transforms.ToTensor()(gt_ldmk_im)
        gt_ldmk = torch.tensor(gt_ldmk)

        return gt_ldmk_im_tensor, gt_ldmk

    def __len__(self):
        return len(self.context_names)


def get_data_loader(root_dir=ROOT_DATASET, K_shots=K_SHOT, workers=NB_WORKERS,
                    loader=LOADER):
    if loader == "json":
        datas = ldmkLoader(root_dir=root_dir, K_shots=K_shots)
    pin = False if DEVICE.type == 'cpu' else True
    train_loader = DataLoader(datas, batch_size=LOAD_BATCH_SIZE_LDMK,
                              shuffle=True, num_workers=workers, pin_memory=pin,
                              drop_last=True)

    return train_loader  # , len(datas.ids)

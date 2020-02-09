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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from settings import (DEVICE, HALF, K_SHOT, LOAD_BATCH_SIZE, LOADER,
                      NB_WORKERS, ROOT_DATASET, ROOT_WEIGHTS, BATCH_SIZE)


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
    return image


def get_landmarks_from_webcam():
    face_landmarks = FaceAlignment(
        landmarks_type=LandmarksType._2D, device="cuda")
    cam = cv2.VideoCapture(0)
    bad_image = True
    while bad_image:
        _, image = cam.read()
        image = cv2.flip(image, 1)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            landmark_pts = face_landmarks.get_landmarks_from_image(
                image)
        image = np.zeros(image.shape, np.float32)
        try:
            landmark_pts = landmark_pts[0]
            write_landmarks_on_image(image, landmark_pts)
            landmark_tensor = transforms.ToTensor()(image)
            bad_image = False
        except TypeError:
            continue
    cam.release()
    torch.cuda.empty_cache()
    return landmark_pts, landmark_tensor.unsqueeze(0).to(DEVICE)


def load_someone():
    if platform.system() == "Windows":
        slash = "\\"
    else:
        slash = "/"
    userid = np.random.choice(glob.glob(f"{ROOT_DATASET}/*"))
    filename = np.random.choice(glob.glob(f"{userid}/*.json"))[:-5]
    id_to_tensor = get_ids()
    itemId = id_to_tensor[filename.split('/')[-2]]

    with open(f"{filename}.json", "r") as file:
        dict_ldmk = json.load(file, object_pairs_hook=lambda x:
                              {int(k): v for k, v in x})

    frames = list(dict_ldmk.keys())

    cvVideo = cv2.VideoCapture(f"{filename}.mp4")

    cvVideo.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
    _, gt_im = cvVideo.read()
    gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB)

    gt_ldmk = dict_ldmk[frames[0]]
    gt_ldmk_im = np.zeros(gt_im.shape, np.float32)
    gt_ldmk_im = write_landmarks_on_image(gt_ldmk_im, gt_ldmk)

    context_tensors_list = []
    for frame in frames:
        cvVideo.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, ctx_img = cvVideo.read()
        ctx_img = cv2.cvtColor(ctx_img, cv2.COLOR_BGR2RGB)
        ctx_ldmk = dict_ldmk[frame]

        ctx_ldmk_img = np.zeros(gt_im.shape, np.float32)
        ctx_ldmk_img = write_landmarks_on_image(ctx_ldmk_img, ctx_ldmk)
        ctx_ldmk_img = transforms.ToTensor()(ctx_ldmk_img)
        ctx_ldmk_img = transforms.Normalize([0, 0, 0],
                                            [255, 255, 225])(ctx_ldmk_img)

        ctx_img = transforms.ToTensor()(ctx_img)
        ctx_img = transforms.Normalize([0.5, 0.5, 0.5],
                                       [0.5, 0.5, 0.5])(ctx_img)
        context_tensors_list.append(ctx_img)
        context_tensors_list.append(ctx_ldmk_img)

    cvVideo.release()

    gt_im_tensor = transforms.ToTensor()(gt_im)
    gt_im_tensor = transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])(gt_im_tensor)
    gt_ldmk_im_tensor = transforms.ToTensor()(gt_ldmk_im)
    context_tensors = torch.cat(context_tensors_list)
    if len(context_tensors.size()) < 4:
        context_tensors.unsqueeze_(0)

    gt_im_tensor = gt_im_tensor.to(DEVICE)
    context_tensors = context_tensors.to(DEVICE)
    itemId = itemId.to(DEVICE)

    return gt_im_tensor, gt_ldmk_im_tensor, context_tensors, itemId


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


class jsonLoader(Dataset):

    def __init__(self, root_dir=ROOT_DATASET, K_shots=K_SHOT):
        super(jsonLoader, self).__init__()
        self.slash = "/"
        if "Windows" in platform.system():
            self.slash = "\\"
        self.K_shots = K_shots
        self.root_dir = root_dir
        print("\tLoading ids...")
        start_time = time.time()
        self.ids = glob.glob(f"{self.root_dir}/*")
        self.id_to_tensor = get_ids()
        print(f"\tIds loaded in {time.time() - start_time}s")
        print("\tLoading videos...")
        start_time = time.time()
        self.context_names = [video[:-5] for video in
                              glob.glob(f"{self.root_dir}/*/*.json")]
        print(f"\tVideos loaded in {time.time() - start_time}s")

    def __getitem__(self, index):
        badLdmks = True
        while badLdmks:
            context_name = self.context_names[index]
            itemId = self.id_to_tensor[context_name.split(self.slash)[-2]]
            with open(f"{context_name}.json", "r") as file:
                dict_ldmk = json.load(file, object_pairs_hook=dictKeytoInt)

            try:
                frames = np.random.choice(list(dict_ldmk.keys()),
                                          self.K_shots + 1)
                badLdmks = False
            except ValueError:
                index = randint(0, len(self.context_names))

        cvVideo = cv2.VideoCapture(f"{context_name}.mp4")

        cvVideo.set(cv2.CAP_PROP_POS_FRAMES, frames[0])
        _, gt_im = cvVideo.read()
        gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB)

        gt_ldmk = dict_ldmk[frames[0]]
        gt_ldmk_im = np.zeros(gt_im.shape, np.float32)
        gt_ldmk_im = write_landmarks_on_image(gt_ldmk_im, gt_ldmk)

        context_tensors_list = []
        for frame in frames[1:]:
            cvVideo.set(cv2.CAP_PROP_POS_FRAMES, frame)
            _, ctx_img = cvVideo.read()
            ctx_img = cv2.cvtColor(ctx_img, cv2.COLOR_BGR2RGB)
            ctx_ldmk = dict_ldmk[frame]

            ctx_ldmk_img = np.zeros(gt_im.shape, np.float32)
            ctx_ldmk_img = write_landmarks_on_image(ctx_ldmk_img, ctx_ldmk)
            ctx_ldmk_img = transforms.ToTensor()(ctx_ldmk_img)
            ctx_ldmk_img = transforms.Normalize([0, 0, 0],
                                                [255, 255, 225])(ctx_ldmk_img)

            ctx_img = transforms.ToTensor()(ctx_img)
            # print("1 ", ctx_img.max(), ctx_img.min())
            ctx_img = transforms.Normalize([0.5, 0.5, 0.5],
                                           [0.5, 0.5, 0.5])(ctx_img)
            # print("2 ", ctx_img.max(), ctx_img.min())
            context_tensors_list.append(ctx_img)
            context_tensors_list.append(ctx_ldmk_img)

        cvVideo.release()
        # print("3 ", gt_im.max(), gt_im.min())
        gt_im_tensor = transforms.ToTensor()(gt_im)
        # print("4 ", gt_im_tensor.max(), gt_im_tensor.min())
        gt_im_tensor = transforms.Normalize([0.5, 0.5, 0.5],
                                            [0.5, 0.5, 0.5])(gt_im_tensor)
        # print("5 ", gt_im_tensor.max(), gt_im_tensor.min())
        gt_ldmk_im_tensor = transforms.ToTensor()(gt_ldmk_im)
        context_tensors = torch.cat(context_tensors_list)
        # print(itemId)
        if HALF:
            gt_im_tensor = gt_im_tensor.half()
            gt_ldmk_im_tensor = gt_ldmk_im_tensor.half()
            context_tensors = context_tensors.half()
        return gt_im_tensor, gt_ldmk_im_tensor, context_tensors, itemId

    def __len__(self):
        return len(self.context_names)


def get_data_loader(root_dir=ROOT_DATASET, K_shots=K_SHOT, workers=NB_WORKERS,
                    loader=LOADER):
    if loader == "json":
        datas = jsonLoader(root_dir=root_dir, K_shots=K_shots)
    pin = False if DEVICE.type == 'cpu' else True
    train_loader = DataLoader(datas, batch_size=LOAD_BATCH_SIZE, shuffle=True,
                              num_workers=workers, pin_memory=pin,
                              drop_last=True)

    return train_loader, len(datas.ids)


def view_batch(loader):
    gt_im, gt_landmarks, context, itemId = next(loader)
    grid = torchvision.utils.make_grid(
        torch.cat((gt_im, gt_landmarks, context), dim=1).view(-1, 3, 224, 224),
        nrow=2 + K_SHOT, padding=2, normalize=True).cpu()
    plt.figure(figsize=(25, 10))
    plt.axis("off")
    plt.title("Training Images exemple\n\nAnchor    Positive  Negative")
    plt.imshow(np.transpose(grid))
    plt.show()

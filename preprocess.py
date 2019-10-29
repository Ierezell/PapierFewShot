import copy
import glob
import json
import os
import platform

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from face_alignment import FaceAlignment, LandmarksType
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import time
from settings import (DEVICE, K_SHOT, LOAD_BATCH_SIZE, ROOT_WEIGHTS,
                      NB_WORKERS, ROOT_DATASET, HALF, LOADER)


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


def get_landmarks_from_webcam(self):
    cam = cv2.VideoCapture(0)
    bad_image = True
    while bad_image:
        _, image = cam.read()
        image = cv2.flip(image, 1)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            landmark_pts = self.face_landmarks.get_landmarks_from_image(
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
    itemId = torch.tensor(int(filename.split(slash)[-2][2:]))

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
        ctx_img = write_landmarks_on_image(ctx_img, ctx_ldmk)
        ctx_img = transforms.ToTensor()(ctx_img)
        context_tensors_list.append(ctx_img)

    cvVideo.release()

    gt_im_tensor = transforms.ToTensor()(gt_im)
    context_tensors = torch.cat(context_tensors_list).unsqueeze(0)
    gt_im_tensor = gt_im_tensor.to(DEVICE)
    context_tensors = context_tensors.to(DEVICE)
    itemId = itemId.to(DEVICE)
    return gt_im_tensor, gt_ldmk, context_tensors, itemId


# #############
# JSON LOADER #
# #############

class jsonLoader(Dataset):
    def get_ids(self):
        # if not os.path.exists(f"{ROOT_WEIGHTS}/ids.json"):
        #     with open(f"{ROOT_WEIGHTS}/ids.json", "w") as file:
        #         json.dump({}, file)

        with open(f"{ROOT_WEIGHTS}ids.json", "w+") as file:
            try:
                json_ids = json.load(file)
            except json.decoder.JSONDecodeError:
                json_ids = {}

        current_id = -1
        id_to_tensor = {}
        for uid in self.ids:
            key = uid.split(self.slash)[-1]
            id_to_tensor[key] = json_ids.get(key, current_id + 1)
            current_id = id_to_tensor[key]

        with open(f"{ROOT_WEIGHTS}/ids.json", "w") as file:
            json.dump(id_to_tensor, file)

        id_to_tensor = {key: torch.tensor(value).view(1)
                        for key, value in id_to_tensor.items()}
        return id_to_tensor

    def __init__(self, root_dir=ROOT_DATASET, K_shots=K_SHOT):
        super(jsonLoader, self).__init__()
        self.slash = "/"
        if "Windows" in platform.system():
            self.slash = "\\"
        self.K_shots = K_shots
        self.root_dir = root_dir
        print("Loading ids...")
        start_time = time.time()
        self.ids = glob.glob(f"{self.root_dir}/*")
        self.id_to_tensor = self.get_ids()
        # self.id_to_tensor = {name.split(self.slash)[-1]:
        #                      torch.tensor(i).view(1)
        #                      for i, name in enumerate(self.ids, start=1)}
        print(f"Ids loaded in {time.time() - start_time}s")
        print("Loading videos...")
        start_time = time.time()
        self.context_names = [video[:-5] for video in
                              glob.glob(f"{self.root_dir}/*/*.json")]
        # print(glob.glob(f"{self.root_dir}\\*\\*.json"))
        # print(self.context_names)
        print(f"videos loaded in {time.time() - start_time}s")

    def __getitem__(self, index):
        context_name = self.context_names[index]
        # itemId = torch.tensor(int(context_name.split(self.slash)[-2][2:]),
        #                       ).view(1)
        itemId = self.id_to_tensor[context_name.split(self.slash)[-2]]

        with open(f"{context_name}.json", "r") as file:
            dict_ldmk = json.load(file,
                                  object_pairs_hook=lambda x: {int(k): v
                                                               for k, v in x})

        frames = np.random.choice(list(dict_ldmk.keys()), self.K_shots + 1)

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
            ctx_img = write_landmarks_on_image(ctx_img, ctx_ldmk)
            ctx_img = transforms.ToTensor()(ctx_img)
            context_tensors_list.append(ctx_img)

        cvVideo.release()

        gt_im_tensor = transforms.ToTensor()(gt_im)
        gt_ldmk_im_tensor = transforms.ToTensor()(gt_ldmk_im)
        context_tensors = torch.cat(context_tensors_list)

        return gt_im_tensor, gt_ldmk_im_tensor, context_tensors, itemId

    def __len__(self):
        return len(self.context_names)


def get_data_loader(root_dir=ROOT_DATASET, K_shots=K_SHOT, workers=NB_WORKERS,
                    loader=LOADER):
    if loader == "json":
        datas = jsonLoader(root_dir=root_dir, K_shots=K_shots)
    elif loader == "frame":
        datas = frameLoader(root_dir=root_dir, K_shots=K_shots)
    # print(len(datas))
    # size_train = int(0.8 * len(datas))
    # size_valid = len(datas) - int(0.8 * len(datas))
    # train_datas, valid_datas = random_split(datas, (size_train, size_valid))
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
    # %matplotlib inline
    plt.figure(figsize=(25, 10))
    plt.axis("off")
    plt.title("Training Images exemple\n\nAnchor    Positive  Negative")
    plt.imshow(np.transpose(grid))
    plt.show()


# ##############
# FRAME LOADER #
# ##############

# class frameLoader(Dataset):
#     def __init__(self, root_dir=ROOT_DATASET, K_shots=K_SHOT):
#         super(frameLoader, self).__init__()
#         self.face_landmarks = FaceAlignment(
#             LandmarksType._2D, device=DEVICE_LANDMARKS)

#         self.K_shots = K_shots
#         self.root_dir = root_dir
#         print("Loading ids...")
#         start_time = time.time()
#         self.ids = glob.glob(f"{self.root_dir}/*")
#         print(f"Ids loaded in {time.time() - start_time}s")
#         print("Loading contexts...")
#         start_time = time.time()
#         self.contexts = glob.glob(f"{self.root_dir}/*/*")
#         print(f"Contexts laoded in {time.time() - start_time}s")
#         # self.mp4files = glob.glob(f"{self.root_dir}/*/*/*")

#         if platform.system() == "Windows":
#             self.id_to_tensor = {name.split("\\")[-1]: torch.tensor(i).view(1)
#                                  for i, name in enumerate(self.ids)}
#         else:
#             self.id_to_tensor = {name.split('/')[-1]: torch.tensor(i).view(1)
#                                  for i, name in enumerate(self.ids)}
#         torch.cuda.empty_cache()

#     def load_random(self, video, total_frame_nb, fusion):
#         badLandmarks = True
#         while badLandmarks:
#             frameIndex = np.random.randint(0, total_frame_nb)
#             video.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
#             _, gt_im = video.read()
#             gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB)
#             with torch.no_grad():
#                 landmarks = self.face_landmarks.get_landmarks_from_image(gt_im)
#             try:
#                 landmarks = landmarks[0]
#                 badLandmarks = False
#             except TypeError:
#                 continue

#         if fusion:
#             image = gt_im
#         else:
#             image = np.zeros(gt_im.shape, np.float32)

#         image = write_landmarks_on_image(image, landmarks)

#         torch.cuda.empty_cache()

#         if fusion:
#             return transforms.ToTensor()(image)
#         else:
#             return transforms.ToTensor()(gt_im), transforms.ToTensor()(image)

#     def __getitem__(self, index):
#         bad_context = True
#         context = self.contexts[index]
#         video_files = glob.glob(f"{context}/*")
#         # print("get !")
#         while bad_context:
#             check_video_files = copy.deepcopy(video_files)
#             # print("Encore un badContext")
#             for v in check_video_files:
#                 try:
#                     cvVideo = cv2.VideoCapture(v)
#                     total_frame_nb = int(cvVideo.get(cv2.CAP_PROP_FRAME_COUNT))
#                     cvVideo.release()

#                     if total_frame_nb < 1:
#                         # print("0 Frames CTX")
#                         raise ValueError

#                 except ValueError:
#                     # print("Bad Video !")
#                     video_files.remove(v)

#             if not video_files:
#                 # print("No video in this context : Loading a new random one.")
#                 context = self.contexts[np.random.randint(len(self.contexts))]
#                 video_files = glob.glob(f"{context}/*")
#                 continue
#             else:
#                 # print("Context ok")
#                 bad_context = False
#         # print("Context bon, je loade")
#         if platform.system() == "Windows":
#             itemId = self.id_to_tensor[context.split("\\")[-2]]
#         else:
#             itemId = self.id_to_tensor[context.split('/')[-2]]

#         if len(video_files) < self.K_shots+1:
#             videos = np.random.choice(video_files, self.K_shots + 1,
#                                       replace=True)
#         else:
#             videos = np.random.choice(video_files, self.K_shots + 1,
#                                       replace=False)
#         # print("N_vidoess")
#         gt_video, *ctx_videos = videos

#         cvVideo = cv2.VideoCapture(gt_video)
#         total_frame_nb = int(cvVideo.get(cv2.CAP_PROP_FRAME_COUNT))
#         gt_im_tensor, gt_landmarks = self.load_random(cvVideo,
#                                                       total_frame_nb,
#                                                       fusion=False)
#         cvVideo.release()

#         # print("Gt ok go for context")
#         context_tensors_list = []
#         for v in ctx_videos:
#             cvVideo = cv2.VideoCapture(v)
#             total_frame_nb = int(cvVideo.get(cv2.CAP_PROP_FRAME_COUNT))
#             context_frame = self.load_random(cvVideo,
#                                              total_frame_nb,
#                                              fusion=True)
#             context_tensors_list.append(context_frame)
#             cvVideo.release()
#         # print("Context ok")
#         context_tensors = torch.cat(context_tensors_list)
#         torch.cuda.empty_cache()

#         context_tensors.requires_grad = True
#         gt_im_tensor.requires_grad = True
#         gt_landmarks.requires_grad = True
#         if HALF:
#             return (gt_im_tensor.half(), gt_landmarks.half(),
#                     context_tensors.half(), itemId)
#         else:
#             return (gt_im_tensor, gt_landmarks, context_tensors, itemId)

#     def __len__(self):
#         return len(self.contexts)

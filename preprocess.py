from face_alignment import FaceAlignment, LandmarksType
from torch.utils.data import DataLoader, Dataset

from settings import (ROOT_DATASET, LOAD_BATCH_SIZE, DEVICE, K_SHOT,
                      DEVICE_LANDMARKS, NB_WORKERS)
import glob
import torch
<<<<<<< HEAD
import platform

=======


>>>>>>> a353fa4a5f7e11fccfe1c06d8f190cc7c482fce8
import cv2
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
import torchvision


class frameLoader(Dataset):
    def __init__(self, root_dir=ROOT_DATASET, K_shots=K_SHOT):
        super(frameLoader, self).__init__()
        self.face_landmarks = FaceAlignment(
            LandmarksType._2D, device=DEVICE_LANDMARKS)

        self.K_shots = K_shots
        self.root_dir = root_dir
        self.ids = glob.glob(f"{self.root_dir}/*")
        self.contexts = glob.glob(f"{self.root_dir}/*/*")
        self.mp4files = glob.glob(f"{self.root_dir}/*/*/*")

<<<<<<< HEAD
        if platform.system()=="Windows":
            self.id_to_tensor = {name.split("\\")[-1]: torch.tensor(i).view(1)
                             for i, name in enumerate(self.ids)}
        else:
            self.id_to_tensor = {name.split('/')[-1]: torch.tensor(i).view(1)
=======
        self.id_to_tensor = {name.split('/')[-1]: torch.tensor(i).view(1)
>>>>>>> a353fa4a5f7e11fccfe1c06d8f190cc7c482fce8
                             for i, name in enumerate(self.ids)}
        torch.cuda.empty_cache()

    def write_landmarks_on_image(self, image, landmarks):
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

    def load_someone(self, limit=200):
        userid = np.random.choice(glob.glob(f"{self.root_dir}/*"))
        context = np.random.choice(glob.glob(f"{userid}/*"))
        mp4file = np.random.choice(glob.glob(f"{context}/*"))
<<<<<<< HEAD
        if platform.system()=="Windows":
            index_user = self.id_to_tensor[mp4file.split("\\")[-3]].to(DEVICE)
        else:
            index_user = self.id_to_tensor[mp4file.split('/')[-3]].to(DEVICE)
=======
        index_user = self.id_to_tensor[mp4file.split('/')[-3]].to(DEVICE)
>>>>>>> a353fa4a5f7e11fccfe1c06d8f190cc7c482fce8
        video = cv2.VideoCapture(mp4file)

        video_continue = True
        context_tensors_list = []
        i = 0
        while True:
            video_continue, image = video.read()
            if not video_continue:
                break
            if i > limit:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks = self.face_landmarks.get_landmarks_from_image(image)
            try:
                landmarks = landmarks[0]
                image = self.write_landmarks_on_image(image, landmarks)
                context_tensors_list.append(transforms.ToTensor()(image))
                i += 1
            except TypeError:
                continue
        video.release()
        all_frames = torch.cat(context_tensors_list).unsqueeze(0).to(DEVICE)
        torch.cuda.empty_cache()
        return all_frames, index_user

    def load_random(self, video, total_frame_nb, fusion):
        badLandmarks = True
        while badLandmarks:
            frameIndex = np.random.randint(0, total_frame_nb)
            video.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
            _, gt_im = video.read()
            gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB)

            landmarks = self.face_landmarks.get_landmarks_from_image(gt_im)
            try:
                landmarks = landmarks[0]
                badLandmarks = False
            except TypeError:
                continue

        if not fusion:
            image = np.zeros(gt_im.shape, np.float32)
        else:
            image = gt_im

        image = self.write_landmarks_on_image(image, landmarks)
        torch.cuda.empty_cache()
        if not fusion:
            return transforms.ToTensor()(gt_im), transforms.ToTensor()(image)
        else:
            return transforms.ToTensor()(image)

    def get_landmarks_from_webcam(self):
        cam = cv2.VideoCapture(0)
        image_ok = False
        while not image_ok:
            _, image = cam.read()
            image = cv2.flip(image, 1)
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks = self.face_landmarks.get_landmarks_from_image(image)
            image = np.zeros(image.shape, np.float32)
            try:
                landmarks = landmarks[0]
                self.write_landmarks_on_image(image, landmarks)
                landmark_tensor = transforms.ToTensor()(image)
                image_ok = True
            except TypeError:
                continue
        cam.release()
        torch.cuda.empty_cache()
        return landmark_tensor.unsqueeze(0).to(DEVICE)
<<<<<<< HEAD

    def __getitem__(self, index):
        mp4File = self.mp4files[index]
        if platform.system()=="Windows":
            itemId = self.id_to_tensor[mp4File.split("\\")[-3]]
        else:
            itemId = self.id_to_tensor[mp4File.split('/')[-3]]

        video = cv2.VideoCapture(mp4File)
        total_frame_nb = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        gt_im_tensor, gt_landmarks = self.load_random(video,
                                                      total_frame_nb,
                                                      fusion=False)

=======

    def __getitem__(self, index):
        mp4File = self.mp4files[index]
        itemId = self.id_to_tensor[mp4File.split('/')[-3]]
        video = cv2.VideoCapture(mp4File)
        total_frame_nb = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        gt_im_tensor, gt_landmarks = self.load_random(video,
                                                      total_frame_nb,
                                                      fusion=False)

>>>>>>> a353fa4a5f7e11fccfe1c06d8f190cc7c482fce8
        context_tensors_list = []
        for _ in range(self.K_shots):
            context_frame = self.load_random(video, total_frame_nb,
                                             fusion=True)
            context_tensors_list.append(context_frame)

        context_tensors = torch.cat(context_tensors_list)
        video.release()

        torch.cuda.empty_cache()
        return gt_im_tensor, gt_landmarks, context_tensors, itemId

    def __len__(self):
        return len(self.mp4files)


def get_data_loader(root_dir=ROOT_DATASET, K_shots=8, workers=NB_WORKERS):
    datas = frameLoader(root_dir=root_dir, K_shots=K_shots)
    # print(len(datas))
    # size_train = int(0.8 * len(datas))
    # size_valid = len(datas) - int(0.8 * len(datas))
    # train_datas, valid_datas = random_split(datas, (size_train, size_valid))
    pin = False if DEVICE.type == 'cpu' else True
    train_loader = DataLoader(datas, batch_size=LOAD_BATCH_SIZE, shuffle=True,
                              num_workers=workers, pin_memory=pin)

    torch.cuda.empty_cache()
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

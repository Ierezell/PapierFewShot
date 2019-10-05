import copy
import glob
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
from settings import (DEVICE, DEVICE_LANDMARKS, K_SHOT, LOAD_BATCH_SIZE,
                      NB_WORKERS, ROOT_DATASET, HALF)


class frameLoader(Dataset):
    def __init__(self, root_dir=ROOT_DATASET, K_shots=K_SHOT):
        super(frameLoader, self).__init__()
        self.face_landmarks = FaceAlignment(
            LandmarksType._2D, device=DEVICE_LANDMARKS)

        self.K_shots = K_shots
        self.root_dir = root_dir
        print("Loading ids...")
        start_time = time.time()
        self.ids = glob.glob(f"{self.root_dir}/*")
        print(f"Ids loaded in {time.time() - start_time}s")
        print("Loading contexts...")
        start_time = time.time()
        self.contexts = glob.glob(f"{self.root_dir}/*/*")
        print(f"Contexts laoded in {time.time() - start_time}s")
        # self.mp4files = glob.glob(f"{self.root_dir}/*/*/*")

        if platform.system() == "Windows":
            self.id_to_tensor = {name.split("\\")[-1]: torch.tensor(i).view(1)
                                 for i, name in enumerate(self.ids)}
        else:
            self.id_to_tensor = {name.split('/')[-1]: torch.tensor(i).view(1)
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

        if platform.system() == "Windows":
            index_user = self.id_to_tensor[mp4file.split("\\")[-3]].to(DEVICE)
        else:
            index_user = self.id_to_tensor[mp4file.split('/')[-3]].to(DEVICE)

        video = cv2.VideoCapture(mp4file)

        context_tensors_list = []
        first_image_landmarks = None
        i = 0
        while True:
            video_continue, image = video.read()
            if (not video_continue) or (i > limit):
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                landmarks = self.face_landmarks.get_landmarks_from_image(image)
            try:
                landmarks = landmarks[0]
                if i == 0:
                    first_image_landmarks = copy.deepcopy(landmarks)
                image = self.write_landmarks_on_image(image, landmarks)
                context_tensors_list.append(transforms.ToTensor()(image))
                i += 1
            except TypeError:
                continue

        video.release()
        all_frames = torch.cat(context_tensors_list).unsqueeze(0).to(DEVICE)

        torch.cuda.empty_cache()
        return all_frames, first_image_landmarks, index_user

    def load_random(self, video, total_frame_nb, fusion):
        badLandmarks = True
        while badLandmarks:
            frameIndex = np.random.randint(0, total_frame_nb)
            video.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
            _, gt_im = video.read()
            gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                landmarks = self.face_landmarks.get_landmarks_from_image(gt_im)
            try:
                landmarks = landmarks[0]
                badLandmarks = False
            except TypeError:
                continue

        if fusion:
            image = gt_im
        else:
            image = np.zeros(gt_im.shape, np.float32)

        image = self.write_landmarks_on_image(image, landmarks)

        torch.cuda.empty_cache()

        if fusion:
            return transforms.ToTensor()(image)
        else:
            return transforms.ToTensor()(gt_im), transforms.ToTensor()(image)

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
                self.write_landmarks_on_image(image, landmark_pts)
                landmark_tensor = transforms.ToTensor()(image)
                bad_image = False
            except TypeError:
                continue
        cam.release()
        torch.cuda.empty_cache()
        return landmark_pts, landmark_tensor.unsqueeze(0).to(DEVICE)

    def __getitem__(self, index):
        bad_context = True
        context = self.contexts[index]
        video_files = glob.glob(f"{context}/*")
        # print("get !")
        while bad_context:
            check_video_files = copy.deepcopy(video_files)
            # print("Encore un badContext")
            for v in check_video_files:
                try:
                    cvVideo = cv2.VideoCapture(v)
                    total_frame_nb = int(cvVideo.get(cv2.CAP_PROP_FRAME_COUNT))
                    cvVideo.release()

                    if total_frame_nb < 1:
                        # print("0 Frames CTX")
                        raise ValueError

                except ValueError:
                    # print("Bad Video !")
                    video_files.remove(v)

            if not video_files:
                # print("No video in this context : Loading a new random one.")
                context = self.contexts[np.random.randint(len(self.contexts))]
                video_files = glob.glob(f"{context}/*")
                continue
            else:
                # print("Context ok")
                bad_context = False
        # print("Context bon, je loade")
        if platform.system() == "Windows":
            itemId = self.id_to_tensor[context.split("\\")[-2]]
        else:
            itemId = self.id_to_tensor[context.split('/')[-2]]

        if len(video_files) < self.K_shots+1:
            videos = np.random.choice(video_files, self.K_shots + 1,
                                      replace=True)
        else:
            videos = np.random.choice(video_files, self.K_shots + 1,
                                      replace=False)
        # print("N_vidoess")
        gt_video, *ctx_videos = videos

        cvVideo = cv2.VideoCapture(gt_video)
        total_frame_nb = int(cvVideo.get(cv2.CAP_PROP_FRAME_COUNT))
        gt_im_tensor, gt_landmarks = self.load_random(cvVideo,
                                                      total_frame_nb,
                                                      fusion=False)
        cvVideo.release()

        # print("Gt ok go for context")
        context_tensors_list = []
        for v in ctx_videos:
            cvVideo = cv2.VideoCapture(v)
            total_frame_nb = int(cvVideo.get(cv2.CAP_PROP_FRAME_COUNT))
            context_frame = self.load_random(cvVideo,
                                             total_frame_nb,
                                             fusion=True)
            context_tensors_list.append(context_frame)
            cvVideo.release()
        # print("Context ok")
        context_tensors = torch.cat(context_tensors_list)
        torch.cuda.empty_cache()
        if HALF:
            return (gt_im_tensor.half(), gt_landmarks.half(),
                    context_tensors.half(), itemId)
        else:
            return (gt_im_tensor, gt_landmarks, context_tensors, itemId)
    # def __getitem__(self, index):
    #     bad_context = True
    #     context = self.contexts[index]
    #     video_files = glob.glob(f"{context}/*")
    #     while bad_context:
    #         if not video_files:
    #             print("No video in this context")
    #             context = self.contexts[np.random.randint(len(self.contexts))]
    #             video_files = glob.glob(f"{context}/*")
    #         else:
    #             if platform.system() == "Windows":
    #                 itemId = self.id_to_tensor[context.split("\\")[-2]]
    #             else:
    #                 itemId = self.id_to_tensor[context.split('/')[-2]]
    #             if len(video_files) < self.K_shots+1:
    #                 videos = np.random.choice(video_files, self.K_shots + 1,
    #                                           replace=True)
    #             else:
    #                 videos = np.random.choice(video_files, self.K_shots + 1,
    #                                           replace=False)
    #             gt_video, *ctx_videos = videos

    #             bad_video = True
    #             cvVideo = None
    #             while bad_video:
    #                 try:
    #                     cvVideo = cv2.VideoCapture(gt_video)
    #                     total_frame_nb = int(
    #                         cvVideo.get(cv2.CAP_PROP_FRAME_COUNT))

    #                     if total_frame_nb == 0:
    #                         print("0 Frame GT")
    #                         cvVideo.release()
    #                         raise ValueError

    #                     gt_im_tensor, gt_landmarks = self.load_random(cvVideo,
    #                                                                   total_frame_nb,
    #                                                                   fusion=False)
    #                     bad_video = False
    #                 except ValueError:
    #                     print("Bad GT Video !")
    #                     video_files.remove(gt_video)

    #                     if not video_files:
    #                         print("No More video")
    #                         bad_video = False
    #                         gt_video = None
    #                     else:
    #                         gt_video = np.random.choice(video_files)

    #             if not gt_video:
    #                 print("No gt swithing context")
    #                 context = self.contexts[np.random.randint(
    #                     len(self.contexts))]
    #                 continue
    #             else:
    #                 cvVideo.release()

    #             context_tensors_list = []
    #             for v in ctx_videos:
    #                 bad_video = True
    #                 while bad_video:
    #                     try:
    #                         cvVideo = cv2.VideoCapture(v)
    #                         total_frame_nb = int(
    #                             cvVideo.get(cv2.CAP_PROP_FRAME_COUNT))
    #                         if total_frame_nb == 0:
    #                             print("0 Frames CTX")
    #                             cvVideo.release()
    #                             raise ValueError

    #                         context_frame = self.load_random(cvVideo,
    #                                                          total_frame_nb,
    #                                                          fusion=True)
    #                         context_tensors_list.append(context_frame)
    #                         bad_video = False

    #                     except ValueError:
    #                         print("Bad CTX Video !")
    #                         video_files.remove(v)
    #                         # if not video_files:
    #                         # bad_video = False
    #                         # else:
    #                         v = np.random.choice(video_files)
    #                 cvVideo.release()
    #             # print("Context ok")
    #             if len(context_tensors_list) != self.K_shots:
    #                 raise AssertionError(f"j'ai pas {self.K_shots} images")
    #                 # continue
    #             # else:
    #             else:
    #                 bad_context = False
    #                 context_tensors = torch.cat(context_tensors_list)
    #             torch.cuda.empty_cache()
    #     return gt_im_tensor, gt_landmarks, context_tensors, itemId

    def __len__(self):
        return len(self.contexts)


def get_data_loader(root_dir=ROOT_DATASET, K_shots=K_SHOT, workers=NB_WORKERS):
    datas = frameLoader(root_dir=root_dir, K_shots=K_shots)
    # print(len(datas))
    # size_train = int(0.8 * len(datas))
    # size_valid = len(datas) - int(0.8 * len(datas))
    # train_datas, valid_datas = random_split(datas, (size_train, size_valid))
    pin = False if DEVICE.type == 'cpu' else True
    train_loader = DataLoader(datas, batch_size=LOAD_BATCH_SIZE, shuffle=True,
                              num_workers=workers, pin_memory=pin,
                              drop_last=True)

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

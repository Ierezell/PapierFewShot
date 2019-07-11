

import glob

import cv2
import numpy as np
import torch
from face_alignment import FaceAlignment, LandmarksType
from matplotlib import pyplot as plt
from torchvision import transforms
from settings import DEVICE
from utils import load_trained_models


def get_landmarks_from_webcam():
    face_landmarks = FaceAlignment(LandmarksType._2D, device="cuda")
    cam = cv2.VideoCapture(0)
    image_ok = False
    while not image_ok:
        _, image = cam.read()
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = face_landmarks.get_landmarks_from_image(image)
        image = np.zeros(image.shape, np.float32)
        try:
            print(landmarks[0][0:17])
            print(landmarks[0][17:22])
            landmarks = landmarks[0]
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

            landmark_tensor = transforms.ToTensor()(image)
            image_ok = True
        except TypeError:
            continue
    cam.release()
    return landmark_tensor.unsqueeze(0).to(DEVICE)


plt.ion()
fig = plt.figure(num='Mon')

while True:
    landmarks = get_landmarks_from_webcam()
    im_landmarks = landmarks[0].detach().cpu().permute(1, 2, 0).numpy()
    plt.imshow(im_landmarks / im_landmarks.max())
    fig.canvas.draw()
    fig.canvas.flush_events()


# print("torch version : ", torch.__version__)
# print("Device : ", DEVICE)
# # torch.autograd.set_detect_anomaly(True)

# embeddings, paramWeights, paramBias = emb(context)
# synth_im = gen(gt_landmarks,  paramWeights, paramBias)

import cv2
import numpy as np
import glob
import torch
from torchvision import transforms
from face_alignment import FaceAlignment, LandmarksType
face_landmarks = FaceAlignment(LandmarksType._2D, device='cpu')

mp4files = glob.glob(f"./dataset/mp4/*/*/*")
ids = glob.glob(f"./dataset/mp4/*")
id_to_tensor = {name.split('/')[-1]: torch.tensor(i, requires_grad=False)
                for i, name in enumerate(ids)}

for index in range(37000):
    torch.set_grad_enabled(False)
    # Load video name
    mp4File = mp4files[index]
    # Get id video
    itemId = id_to_tensor[mp4File.split('/')[-3]]
    # Load video
    video = cv2.VideoCapture(mp4File)
    print("Frames : ", video.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get total frame video
    total_frame_nb = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get random frame nb
    frameIndex = np.random.randint(0, total_frame_nb)
    # Set random frame nb
    video.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
    # Get image
    _, gt_im = video.read()
    # To tensor without grad
    gt_im_tensor = transforms.ToTensor()(gt_im)
    gt_im_tensor.requires_grad = False

    # Get landmarks
    landmarks = face_landmarks.get_landmarks_from_image(gt_im)[0]
    # Create an image with it
    landmark_image = np.zeros(gt_im.shape, np.float32)
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

    gt_landmarks = transforms.ToTensor()(landmark_image)
    gt_landmarks.requires_grad = False
    # Get random context index
    contextIndex = (frameIndex+(total_frame_nb//10)) % total_frame_nb
    # Reduce it if necessary to get 8 frames
    indexOk = False
    while not indexOk:
        if contextIndex + 8 > total_frame_nb:
            contextIndex -= 1
        else:
            indexOk = True
    # Set context index
    video.set(cv2.CAP_PROP_POS_FRAMES, contextIndex)

    # Context tensor (list)
    context = torch.Tensor()
    context.requires_grad = False
    for i in range(8):
        _, contextFrame = video.read()
        context_tensor = transforms.ToTensor()(contextFrame)
        context_tensor.requires_grad = False
        context = torch.cat((context, context_tensor))
        context.requires_grad = False

    video.release()

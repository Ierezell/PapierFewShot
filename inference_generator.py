import cv2
import numpy as np
import torch
from face_alignment import FaceAlignment, LandmarksType
from matplotlib import pyplot as plt
from torchvision import transforms
from preprocess import get_data_loader, load_someone, write_landmarks_on_image
from utils import load_models
from settings import DEVICE

nb_pers = torch.load("./weights/top_small/Discriminator.pt",
                     map_location=DEVICE).get("embeddings.weight").size(0)

emb, gen, disc = load_models(nb_pers, load_previous_state=False, model="small",
                             root_path_weights="./weights/top_small/",
                             freeze=True)

gt_im_tensor, gt_ldmk, context_tensors, itemId = load_someone()

real_image = gt_im_tensor.cpu().permute(1, 2, 0).numpy()

print(context_tensors.size(1)/6, "  Frames Loaded")
face_landmarks = FaceAlignment(landmarks_type=LandmarksType._2D, device="cuda")
plt.ion()
cam = cv2.VideoCapture(0)

with torch.no_grad():
    embeddings, paramWeights, paramBias, layersUp = emb(context_tensors)

first = True
while True:
    _, image = cam.read()
    image = cv2.flip(image, 1)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        landmark_pts = face_landmarks.get_landmarks_from_image(image)
    landmark_pts = landmark_pts[0]
    landmarks_img = np.zeros(image.shape, np.float32)
    # image = write_landmarks_on_image(image, landmark_pts)
    landmarks_img = write_landmarks_on_image(landmarks_img, landmark_pts)
    landmark_tensor = transforms.ToTensor()(landmarks_img)
    landmark_tensor = landmark_tensor.unsqueeze(0).to("cuda")
    with torch.no_grad():
        synth_im = gen(landmark_tensor, paramWeights, paramBias, layersUp)
        score_synth, _ = disc(torch.cat((synth_im, landmark_tensor), dim=1),
                              itemId)
    score_synth = float(score_synth.data.cpu().numpy())

    im_synth = synth_im.squeeze().detach().cpu().permute(1, 2, 0).numpy()
    im_synth = (im_synth + 1) / 2

    print(im_synth.max(), im_synth.min())
    print(im_synth.shape)
    fig, axes = plt.subplots(2, 2, num='Inf')
    axes[0, 0].clear()
    axes[0, 1].clear()
    axes[1, 0].clear()
    axes[0, 0].imshow(im_synth/im_synth.max())
    axes[0, 1].imshow(landmarks_img/landmarks_img.max())
    axes[1, 0].imshow(image)
    if first:
        axes[1, 1].imshow(real_image)
        first = False

    print(score_synth)

    fig.canvas.draw()
    fig.canvas.flush_events()
cam.release()

from matplotlib import pyplot as plt
import cv2
from face_alignment import FaceAlignment, LandmarksType
import torch
import numpy as np

plt.ion()
fig, axes = plt.subplots(2, 2, num='Metrics')
cam = cv2.VideoCapture(0)
face_landmarks = FaceAlignment(LandmarksType._2D, device="cuda")

ref_ldmk = np.ones((224, 224), np.float32)
ref_img = np.ones((224, 224), np.float32)
ref_ldmk_pts = np.ones((68, 2), np.float32)

while True:
    _, image = cam.read()
    image = cv2.flip(image, 1)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        landmark_pts = face_landmarks.get_landmarks_from_image(image)
    ldmk = np.zeros(image.shape, np.float32)
    try:
        landmark_pts = landmark_pts[0]
        landmark_pts[:, 0] = landmark_pts[:, 0] - min(landmark_pts[:, 0])
        landmark_pts[:, 1] = landmark_pts[:, 1] - min(landmark_pts[:, 1])
        landmark_pts[:, 0] = (landmark_pts[:, 0] / max(landmark_pts[:, 1]))*224
        landmark_pts[:, 1] = (landmark_pts[:, 1] / max(landmark_pts[:, 1]))*224
        # Machoire
        cv2.polylines(ldmk, [np.int32(landmark_pts[0:17])],
                      isClosed=False, color=(0, 255, 0))
        # Sourcil Gauche
        cv2.polylines(ldmk, [np.int32(landmark_pts[17:22])],
                      isClosed=False, color=(255, 0, 0))
        # Sourcil droit
        cv2.polylines(ldmk, [np.int32(landmark_pts[22:27])],
                      isClosed=False, color=(255, 0, 0))
        # Nez arrete
        cv2.polylines(ldmk, [np.int32(landmark_pts[27:31])],
                      isClosed=False, color=(255, 0, 255))
        # Nez narine
        cv2.polylines(ldmk, [np.int32(landmark_pts[31:36])],
                      isClosed=False, color=(255, 0, 255))
        # Oeil gauche
        cv2.polylines(ldmk, [np.int32(landmark_pts[36:42])],
                      isClosed=True, color=(0, 0, 255))
        # oeil droit
        cv2.polylines(ldmk, [np.int32(landmark_pts[42:48])],
                      isClosed=True, color=(0, 0, 255))
        # Bouche exterieur
        cv2.polylines(ldmk, [np.int32(landmark_pts[48:60])],
                      isClosed=True, color=(255, 255, 0))
        # Bouche interieur
        cv2.polylines(ldmk, [np.int32(landmark_pts[60:68])],
                      isClosed=True, color=(255, 255, 0))

        axes[0, 0].clear()
        axes[0, 1].clear()
        axes[1, 0].clear()
        axes[1, 1].clear()
        axes[0, 0].imshow(ldmk / ldmk.max())
        axes[0, 1].imshow(image / image.max())
        axes[1, 0].imshow(ref_ldmk / ref_ldmk.max())
        axes[1, 1].imshow(ref_img / ref_img.max())
        axes[0, 0].text(0, 0, f"{np.linalg.norm(ref_ldmk_pts-landmark_pts)}")
        fig.canvas.draw()
        fig.canvas.flush_events()

        # if (cv2.waitKey(0) & 0xFF) == ord('z'):
        #     print("New ref !")
        #     ref_img = image
        #     ref_ldmk = ldmk
        #     ref_ldmk_pts = landmark_pts

        if plt.waitforbuttonpress(0.001):
            print("New reference !")
            ref_img = image
            ref_ldmk = ldmk
            ref_ldmk_pts = landmark_pts

    except TypeError:
        continue
cam.release()


# print("torch version : ", torch.__version__)
# print("Device : ", DEVICE)
# # torch.autograd.set_detect_anomaly(True)

# embeddings, paramWeights, paramBias = emb(context)
# synth_im = gen(gt_landmarks,  paramWeights, paramBias)

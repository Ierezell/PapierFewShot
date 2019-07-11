

import torch
from matplotlib import pyplot as plt
from settings import ROOT_DATASET
from utils import load_trained_models
from preprocess import frameLoader

frameloader = frameLoader(ROOT_DATASET, 0)

emb, gen, disc = load_trained_models(len(frameloader.ids))

context, user_id = frameloader.load_someone(limit=2000)

real_image = context[0].narrow(0, 0, 3).cpu().permute(1, 2, 0).numpy()

print(context.size(1)/3, "  Frames Loaded")

plt.ion()

with torch.no_grad():
    embeddings, paramWeights, paramBias = emb(context)

    while True:
        landmarks = frameloader.get_landmarks_from_webcam()
        synth_im = gen(landmarks, paramWeights, paramBias)
        score_synth, _ = disc(torch.cat((synth_im, landmarks), dim=1), user_id)

        im_synth = synth_im[0].detach().cpu().permute(1, 2, 0).numpy()
        im_landmarks = landmarks[0].detach().cpu().permute(1, 2, 0).numpy()
        fig, axes = plt.subplots(2, 2, num='Inf')
        axes[0, 0].imshow(im_synth / im_synth.max())
        axes[0, 1].imshow(im_landmarks / im_landmarks.max())
        axes[1, 0].imshow(real_image / real_image.max())

        print(score_synth)

        fig.canvas.draw()
        fig.canvas.flush_events()


# print("torch version : ", torch.__version__)
# print("Device : ", DEVICE)
# # torch.autograd.set_detect_anomaly(True)

# embeddings, paramWeights, paramBias = emb(context)
# synth_im = gen(gt_landmarks,  paramWeights, paramBias)

import torch
from matplotlib import pyplot as plt
from utils import load_models
from preprocess import load_someone, get_landmarks_from_webcam

emb, gen, disc = load_models(1)

gt_im_tensor, gt_ldmk, context_tensors, itemId = load_someone()

real_image = gt_im_tensor.cpu().permute(1, 2, 0).numpy()

print(context_tensors.size(1)/6, "  Frames Loaded")

plt.ion()

with torch.no_grad():
    embeddings, paramWeights, paramBias, layersUp = emb(context_tensors)

    while True:
        ldm_pts, landmarks_img = get_landmarks_from_webcam()
        synth_im = gen(landmarks_img, paramWeights, paramBias, layersUp)
        score_synth, _ = disc(
            torch.cat((synth_im, landmarks_img), dim=1), itemId)

        im_synth = synth_im[0].detach().cpu().permute(1, 2, 0).numpy()
        im_landmarks = landmarks_img[0].detach().cpu().permute(1, 2, 0).numpy()
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

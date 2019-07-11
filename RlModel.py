from preprocess import frameLoader
from utils import load_trained_models
from settings import DEVICE, ROOT_DATASET
from matplotlib import pyplot as plt

from numpy import random
from settings import GAMMA, EPS_DECAY, EPS_END, EPS_START
from torch import nn
from losses import vgg_face_dag
import torch
import math


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = 2622
        self.action_space = 68*4

        self.repres_image = vgg_face_dag()
        self.l1 = nn.Linear(self.state_space, 512, bias=False)
        self.l2 = nn.Linear(512, self.action_space, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU()
        self.gamma = GAMMA
        self.steps_done = 0

    def forward(self, image):
        out = self.repres_image(image)
        out = self.l1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.softmax(out)
        return out

    def select_action(self, state, policy):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            return policy(state).data.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_space)]],
                                device=DEVICE, dtype=torch.long)


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

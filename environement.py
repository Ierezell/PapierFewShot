
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from face_alignment import FaceAlignment, LandmarksType
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from preprocess import frameLoader
from settings import (DEVICE, MAX_DEQUE_LANDMARKS, MAX_ITER_PERSON, MODEL,
                      PRINT_EVERY, DEVICE_LANDMARKS)
from utils import load_models


class Environement:
    def __init__(self):
        self.landmarks_creator = FaceAlignment(LandmarksType._2D,
                                               device=DEVICE_LANDMARKS)
        self.landmarks = None

        self.frameloader = frameLoader()

        (self.embedder,
         self.generator,
         self.discriminator) = load_models(len(self.frameloader.ids))
        self.embedder = self.embedder.eval()
        self.generator = self.generator.eval()
        self.discriminator = self.discriminator.eval()

        self.landmarks_done = deque(maxlen=MAX_DEQUE_LANDMARKS)
        self.contexts = None
        self.user_ids = None
        self.embeddings = None
        self.paramWeights = None
        self.paramBias = None
        self.layersUp = None
        self.iterations = 0
        self.episodes = 0
        self.max_iter = MAX_ITER_PERSON
        self.fig, self.axes = plt.subplots(2, 2)

        self.writer = SummaryWriter()

    def new_person(self):
        torch.cuda.empty_cache()
        self.landmarks_done = deque(maxlen=10)

        (self.contexts,
         self.landmarks,
         self.user_ids) = self.frameloader.load_someone(limit=200)

        if MODEL == "big":
            with torch.no_grad():
                (self.embeddings,
                 self.paramWeights,
                 self.paramBias,
                 self.layersUp) = self.embedder(self.contexts)

        elif MODEL == "small":
            with torch.no_grad():
                (self.embeddings,
                 self.paramWeights,
                 self.paramBias,
                 self.layersUp) = self.embedder(self.contexts)

        self.iterations = 0
        self.episodes += 1
        self.synth_im = self.contexts.narrow(1, 0, 3)
        synth_im = self.synth_im[0].cpu().permute(1, 2, 0).numpy()

        self.axes[0, 0].clear()
        self.axes[0, 0].imshow(synth_im/synth_im.max())
        self.axes[0, 0].axis("off")
        self.axes[0, 0].set_title('State')

        self.axes[1, 0].clear()
        self.axes[1, 0].imshow(synth_im/synth_im.max())
        self.axes[1, 0].axis("off")
        self.axes[1, 0].set_title('Ref')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        torch.cuda.empty_cache()

    def step(self, action):
        self.iterations += 1
        done = False
        if self.iterations > self.max_iter:
            done = True
        point_nb = action % 68
        type_action = action // 68

        if type_action == 0:
            self.landmarks[point_nb][0] += 5
        if type_action == 1:
            self.landmarks[point_nb][0] -= 5
        if type_action == 2:
            self.landmarks[point_nb][1] += 5
        if type_action == 3:
            self.landmarks[point_nb][1] -= 5

        reward = self.get_reward()
        self.writer.add_scalar("reward", reward,
                               global_step=self.iterations*self.episodes)

        if self.iterations % PRINT_EVERY == 0:
            self.writer.add_figure("Fig", self.fig,
                                   global_step=self.iterations*self.episodes)
        return self.synth_im, reward, done

    def get_reward(self):
        landmarks_img = self.frameloader.write_landmarks_on_image(
            np.zeros((224, 224, 3), dtype=np.float32), self.landmarks)

        self.landmarks_img = transforms.ToTensor()(landmarks_img)
        self.landmarks_img = self.landmarks_img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            self.synth_im = self.generator(self.landmarks_img,
                                           self.paramWeights,
                                           self.paramBias, self.layersUp)
        print(landmarks_img.min(), landmarks_img.max())
        print(synth_im.min(), synth_im.max())
        self.axes[0, 1].clear()
        self.axes[0, 1].imshow(landmarks_img/landmarks_img.max())
        self.axes[0, 1].axis("off")
        self.axes[0, 1].set_title('Landmarks (latent space)')

        synth_im = self.synth_im[0].detach().cpu().permute(1, 2, 0).numpy()
        self.axes[0, 0].clear()
        self.axes[0, 0].imshow(synth_im / synth_im.max())
        self.axes[0, 0].axis("off")
        self.axes[0, 0].set_title('State')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # print("self.synth_im", self.synth_im.size())
        # print("self.landmarks_img", self.landmarks_img.size())
        # print(" self.user_ids", self.user_ids.size())
        with torch.no_grad():
            score_disc, _ = self.discriminator(torch.cat((self.synth_im,
                                                          self.landmarks_img),
                                                         dim=1),
                                               self.user_ids)
        if self.landmarks in self.landmarks_done:
            score_redoing = -100
        else:
            score_redoing = 0
            self.landmarks_done.append(self.landmarks)

        for point in range(0, 68):
            if (self.landmarks[point][0] < 0 or
                self.landmarks[point][0] > 224 or
                self.landmarks[point][1] < 0 or
                    self.landmarks[point][1] > 224):
                score_outside = -50
                break
        else:
            score_outside = 0
        # print("score_disc : ", score_disc)
        # print("score_redoing : ", score_redoing)
        # print("score_outside : ", score_outside)
        # print("Score Tot : ", score_disc/10 + score_redoing + score_outside)
        # print("\n")
        return score_disc/10 + score_redoing + score_outside

    def reset(self):
        self.landmarks = self.begining_landmarks.copy()
        self.landmarks_done = deque(maxlen=1000)
        self.contexts = None
        self.user_ids = None
        self.embeddings = None
        self.paramWeights = None
        self.paramBias = None
        self.layersUp = None
        self.iterations = 0
        self.episodes = 0
        self.writer = SummaryWriter()
        # torch.cuda.empty_cache()

    def finish(self):
        self.writer.close()
        # torch.cuda.empty_cache()

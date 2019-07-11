
import numpy as np

from settings import ROOT_DATASET
from utils import load_trained_models
from preprocess import frameLoader
import torch
from collections import deque


class Environement:
    def __init__(self):
        self.begining_landmarks = [[39.,  80.],
                                   [42., 101.],
                                   [47., 120.],
                                   [50., 138.],
                                   [55., 157.],
                                   [65., 172.],
                                   [76., 183.],
                                   [87., 196.],
                                   [108., 207.],
                                   [129., 196.],
                                   [139., 180.],
                                   [147., 167.],
                                   [155., 149.],
                                   [160., 128.],
                                   [163., 109.],
                                   [166.,  91.],
                                   [166.,  70.],
                                   [47.,  46.],
                                   [52.,  33.],
                                   [60.,  27.],
                                   [71.,  25.],
                                   [81.,  27.],
                                   [123.,  25.],
                                   [131.,  20.],
                                   [142.,  20.],
                                   [152.,  22.],
                                   [158.,  35.],
                                   [102.,  64.],
                                   [100.,  80.],
                                   [102.,  96.],
                                   [102., 109.],
                                   [92., 122.],
                                   [97., 125.],
                                   [102., 125.],
                                   [110., 122.],
                                   [116., 122.],
                                   [60.,  67.],
                                   [65.,  62.],
                                   [73.,  62.],
                                   [81.,  67.],
                                   [76.,  70.],
                                   [65.,  70.],
                                   [121.,  62.],
                                   [131.,  59.],
                                   [142.,  56.],
                                   [147.,  62.],
                                   [139.,  64.],
                                   [131.,  64.],
                                   [81., 159.],
                                   [89., 151.],
                                   [100., 146.],
                                   [105., 146.],
                                   [110., 146.],
                                   [121., 149.],
                                   [129., 157.],
                                   [121., 164.],
                                   [113., 170.],
                                   [108., 172.],
                                   [97., 172.],
                                   [89., 167.],
                                   [84., 157.],
                                   [100., 154.],
                                   [105., 154.],
                                   [113., 154.],
                                   [126., 157.],
                                   [113., 159.],
                                   [105., 159.],
                                   [100., 159.]]
        self.landmarks = self.begining_landmarks.copy()
        self.frameloader = frameLoader(ROOT_DATASET, 0)

        (self.embedder,
         self.generator,
         self.discriminator) = load_trained_models(len(self.frameloader.ids))

        self.landmarks_done = deque(maxlen=10000)
        self.context = None
        self.user_id = None
        self.embeddings = None
        self.paramWeights = None
        self.paramBias = None
        self.iterations = 0
        self.episodes = 0
        self.max_iter = 2000000

    def new_person(self):
        self.context, self.user_id = self.frameloader.load_someone(limit=2000)
        (self.embeddings,
         self.paramWeights,
         self.paramBias) = self.emb(self.context)
        self.iterations = 0
        self.episodes += 1

    def step(self, action):
        self.iterations += 1
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
        return self.landmarks, reward, done

    def get_reward(self):
        self.landmarks_img = self.frameloader.write_landmarks_on_image(
            np.zeros((224, 224), dtype=np.float32), self.landmarks)

        self.synth_im = self.generator(self.landmarks_img)

        score_disc = self.discriminator(torch.cat((self.synth_im,
                                                   self.landmarks_img), dim=1),
                                        self.user_id)

        if self.landmarks in self.landmarks_done:
            score_redoing = -100
        else:
            score_redoing = 0
            self.landmarks_done.append(self.landmarks)
        return score_disc/10 + score_redoing

    def reset(self):
        self.landmarks = self.begining_landmarks.copy()
        self.landmarks_done = deque(maxlen=1000)
        self.context = None
        self.user_id = None
        self.embeddings = None
        self.paramWeights = None
        self.paramBias = None
        self.iterations = 0
        self.episodes = 0

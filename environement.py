
from torchvision import transforms
import numpy as np

from settings import BATCH_SIZE, DEVICE, MODEL
from utils import load_trained_models
from preprocess import frameLoader
import torch
from collections import deque
import matplotlib.pyplot as plt


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
        self.frameloader = frameLoader()

        (self.embedder,
         self.generator,
         self.discriminator) = load_trained_models(len(self.frameloader.ids))
        self.landmarks_done = deque(maxlen=10000)
        self.contexts = None
        self.user_ids = None
        self.embeddings = None
        self.paramWeights = None
        self.paramBias = None
        self.layersUp = None
        self.iterations = 0
        self.episodes = 0
        self.max_iter = 2000000
        self.fig, self.axes = plt.subplots(2, 2)
        torch.cuda.empty_cache()

    def new_person(self):
        # self.contexts = torch.tensor(np.zeros(BATCH_SIZE, dtype=np.float),
        #                              dtype=torch.float, device="cuda")
        # self.user_ids = torch.tensor(np.zeros(BATCH_SIZE, dtype=np.float),
        #                              dtype=torch.float, device="cuda")
        # for b in range(BATCH_SIZE):
        #     context, user_id = self.frameloader.load_someone(limit=2000)
        #     print(context.size())
        #     print(self.contexts.size())

        #     self.contexts = torch.cat((self.contexts, context), dim=0)
        #     self.user_ids = torch.cat((self.user_ids, user_id), dim=0)
        # input("going to load someone")
        self.contexts, self.user_ids = self.frameloader.load_someone(limit=20)
        if MODEL == "big":
            (self.embeddings,
             self.paramWeights,
             self.paramBias,
             self.layersUp) = self.embedder(self.contexts)
        elif MODEL == "small":
            (self.embeddings,
             self.paramWeights,
             self.paramBias) = self.embedder(self.contexts)

        self.iterations = 0
        self.episodes += 1
        self.synth_im = self.contexts.narrow(1, 0, 3)

        self.axes[0, 0].clear()
        synth_im = self.synth_im[0].cpu().permute(1, 2, 0).numpy()
        self.axes[0, 0].imshow(synth_im/synth_im.max())
        self.axes[0, 0].axis("off")
        self.axes[0, 0].set_title('State')

        self.axes[1, 0].clear()
        synth_im = self.synth_im[0].cpu().permute(1, 2, 0).numpy()
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
        return self.synth_im, reward, done

    def get_reward(self):
        landmarks_img = self.frameloader.write_landmarks_on_image(
            np.zeros((224, 224, 3), dtype=np.float32), self.landmarks)

        self.landmarks_img = transforms.ToTensor()(landmarks_img)
        self.landmarks_img = self.landmarks_img.unsqueeze(0).to(DEVICE)
        if MODEL == "big":
            self.synth_im = self.generator(self.landmarks_img,
                                           self.paramWeights,
                                           self.paramBias, self.layersUp)
        elif MODEL == "small":
            self.synth_im = self.generator(self.landmarks_img,
                                           self.paramWeights,
                                           self.paramBias)

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
        score_disc, _ = self.discriminator(torch.cat((self.synth_im,
                                                      self.landmarks_img), dim=1),
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

        return score_disc/10 + score_redoing + score_outside

    def reset(self):
        self.landmarks = self.begining_landmarks.copy()
        self.landmarks_done = deque(maxlen=1000)
        self.contexts = None
        self.user_ids = None
        self.embeddings = None
        self.paramWeights = None
        self.paramBias = None
        self.iterations = 0
        self.episodes = 0
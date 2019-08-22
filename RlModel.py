
from settings import GAMMA
from torch import nn
from losses import vgg_face_dag
from collections import deque
import numpy as np
import torch

<<<<<<< HEAD
<<<<<<< HEAD
# TODO LSTM dans la politique

=======
>>>>>>> d23d6bbcfb8c1d6a94c0b9fc5cb92bb806ed1a86
=======
>>>>>>> d23d6bbcfb8c1d6a94c0b9fc5cb92bb806ed1a86

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = 2622
        self.action_space = 68*4

        self.repres_image = vgg_face_dag()
        self.repres_image = self.repres_image.eval()
        for name, param in self.repres_image.named_parameters():
            param.requires_grad = False
<<<<<<< HEAD
<<<<<<< HEAD

        grad_param_vgg = sum([np.prod(p.size()) if p.requires_grad else 0
                              for p in self.repres_image.parameters()])
        print("Nombre de paramètres vggface: ", f"{grad_param_vgg:,}")
=======
        print("Nombre de paramètres vggface: ",
              f"{sum([np.prod(p.size()) if p.requires_grad else 0 for p in self.repres_image.parameters()]):,}")
>>>>>>> d23d6bbcfb8c1d6a94c0b9fc5cb92bb806ed1a86
=======
        print("Nombre de paramètres vggface: ",
              f"{sum([np.prod(p.size()) if p.requires_grad else 0 for p in self.repres_image.parameters()]):,}")
>>>>>>> d23d6bbcfb8c1d6a94c0b9fc5cb92bb806ed1a86
        self.l1 = nn.Linear(self.state_space, 512, bias=False)
        self.l2 = nn.Linear(512, self.action_space, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU()
        self.gamma = GAMMA
        self.steps_done = 0
<<<<<<< HEAD
<<<<<<< HEAD
        self.replay_memory = deque(maxlen=10000)
=======
        # self.replay_memory = deque(maxlen=10000)
>>>>>>> d23d6bbcfb8c1d6a94c0b9fc5cb92bb806ed1a86
=======
        # self.replay_memory = deque(maxlen=10000)
>>>>>>> d23d6bbcfb8c1d6a94c0b9fc5cb92bb806ed1a86
        torch.cuda.empty_cache()

    def forward(self, image):
        with torch.no_grad():
            out = self.repres_image(image)
        out = self.l1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.l2(out)
        out = self.softmax(out)
        return out

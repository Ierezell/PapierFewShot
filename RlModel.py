
from collections import deque

import numpy as np
import torch
from torch import nn

from losses import vgg_face_dag
from settings import GAMMA


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = 2622
        self.action_space = 68*4

        self.repres_image = vgg_face_dag(freeze=False)
        for name, param in self.repres_image.named_parameters():
            if "fc" in name:
                print(name)
                param.requires_grad = False
            else:
                param.requires_grad = False

        grad_param_vgg = sum([np.prod(p.size()) if p.requires_grad else 0
                              for p in self.repres_image.parameters()])
        print("Nombre de paramètres vggface: ", f"{grad_param_vgg:,}")

        # self.l1 = nn.Linear(self.state_space, 512, bias=False)
        self.rnn = nn.LSTM(input_size=self.state_space, hidden_size=512,
                           num_layers=2, bias=True, batch_first=True,
                           dropout=0.2, bidirectional=False)
        self.l2 = nn.Linear(512, self.action_space, bias=False)
        self.prev_img_repr = deque(maxlen=10)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU()
        self.gamma = GAMMA
        self.steps_done = 0

        self.replay_memory = deque(maxlen=1000)
        torch.cuda.empty_cache()

    def forward(self, image):
        repr_img = self.repres_image(image)
        # print("repr_img : ", repr_img.size())

        self.prev_img_repr.append(repr_img)
        # print("prev_img_repr : ", self.prev_img_repr)

        tensor_img_repr = torch.cat(list(self.prev_img_repr)).unsqueeze(dim=0)
        # tensor_img_repr=torch.nn.utils.rnn.pack_sequence(self.prev_img_repr)
        # print("tensor_img_repr : ", tensor_img_repr.size())

        out_rnn, (hidden, cells) = self.rnn(tensor_img_repr)
        # out_rnn,len_out_rnn=torch.nn.utils.rnn.pad_packed_sequence(out_rnn)
        # out_rnn = out_rnn.view(1, 512)
        # print("len_out_rnn : ", len_out_rnn)
        # print("out_rnn : ", out_rnn.size())
        # print("hidden : ", hidden.size())
        hidden = torch.sum(hidden, dim=0)
        # print("hidden : ", hidden.size())
        out_rnn_relu = self.relu(hidden)

        out_linear = self.l2(out_rnn_relu)
        # print("out_linear : ", out_linear.size())

        out = self.dropout(out_linear)
        probas = self.softmax(out)
        # print("probas : ", probas.size())
        return probas

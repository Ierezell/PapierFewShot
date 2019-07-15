

import glob

import cv2
import numpy as np
import torch
from face_alignment import FaceAlignment, LandmarksType
from matplotlib import pyplot as plt
from torchvision import transforms
from settings import (DEVICE, ROOT_DATASET, LEARNING_RATE_RL,
                      EPS_DECAY, EPS_END, EPS_START)
from utils import load_trained_models
from preprocess import frameLoader
from environement import Environement
from RlModel import Policy
from torch.optim import Adam
from torch import nn
import random
import math
plt.ion()

environement = Environement()
torch.cuda.empty_cache()
environement.new_person()
torch.cuda.empty_cache()
policy = Policy()
policy = policy.to(DEVICE)
torch.cuda.empty_cache()
print("Nombre de paramètres police: ",
      f"{sum([np.prod(p.size()) if p.requires_grad else 0 for p in policy.parameters()]):,}")

optimizer = Adam(policy.parameters(), lr=LEARNING_RATE_RL)


# ################################################

# ################################################
# ################################################
# ################################################
# ################################################
# ################################################
# ################################################
# ################################################


iteration = 0
done = False
criterion = nn.MSELoss()
while not done:
    print("k")
    state = environement.synth_im
    # print(state.size())
    probas = policy(state)
    # Choose action
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * iteration / EPS_DECAY)
    iteration += 1
    if sample > eps_threshold:
        action_index = torch.argmax(probas)
    else:
        action_index = torch.randint(low=0, high=policy.action_space,
                                     size=(1,), dtype=torch.int)
    # Apply action
    # action_index = torch.tensor([32])
    print(action_index)
    new_state, reward, done = environement.step(action_index)
    state = new_state
    # policy.replay_memory.append((state, action_index, reward, new_state, done))
    # minibatch = random.sample(policy.replay_memory,
    #   min(len(policy.replay_memory), 8))

    # # unpack minibatch
    # state_batch = torch.cat(tuple(d[0] for d in minibatch))
    # action_batch = torch.cat(tuple(d[1] for d in minibatch))
    # reward_batch = torch.cat(tuple(d[2] for d in minibatch))
    # state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

    # if torch.cuda.is_available():  # put on GPU if CUDA is available
    #     state_batch = state_batch.cuda()
    #     action_batch = action_batch.cuda()
    #     reward_batch = reward_batch.cuda()
    #     state_1_batch = state_1_batch.cuda()

    # # get output for the next state
    # output_1_batch = policy(state_1_batch)

    # # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
    # y_batch = torch.cat(
    #     tuple(reward_batch[i] if minibatch[i][4]
    #           else reward_batch[i]+policy.gamma*torch.max(output_1_batch[i])
    #           for i in range(len(minibatch))
    #           )
    # )

    # # extract Q-value
    # q_value = torch.sum(policy(state_batch) * action_batch, dim=1)

    # # PyTorch accumulates gradients by default,
    # # so they need to be reset in each pass
    # optimizer.zero_grad()

    # # returns a new Tensor, detached from the current graph,
    # # the result will never require gradient
    # y_batch = y_batch.detach()

    # # calculate loss
    # loss = criterion(q_value, y_batch)

    # # do backward pass
    # loss.backward()
    # optimizer.step()

    # # set state to be state_1
    # state = new_state

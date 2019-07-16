

import glob

import cv2
import numpy as np
import torch
from face_alignment import FaceAlignment, LandmarksType
from matplotlib import pyplot as plt
from torchvision import transforms
from settings import (DEVICE, ROOT_DATASET, LEARNING_RATE_RL,
                      EPS_DECAY, EPS_END, EPS_START, BATCH_SIZE_RL)
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
    # print("k")
    state = environement.synth_im
    # print(state.size())
    probas = policy(state)
    # Choose action
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * iteration / EPS_DECAY)
    iteration += 1
    if sample > eps_threshold:
        action_index = torch.argmax(probas).unsqueeze(-1).int()
        print("e : ", action_index, action_index.type(), action_index.device)
    else:
        action_index = torch.randint(low=0, high=policy.action_space,
                                     size=(1,), dtype=torch.int, device="cuda")
        print("p : ", action_index, action_index.type(), action_index.device)
    # Apply action
    # action_index = torch.tensor([32])
    new_state, reward, done = environement.step(action_index)
    policy.replay_memory.append((state.detach().cpu(),
                                 action_index.detach().cpu(),
                                 reward.detach().cpu(),
                                 new_state.detach().cpu(),
                                 done))
    minibatch = random.sample(policy.replay_memory,
                              min(len(policy.replay_memory), BATCH_SIZE_RL))
    # print(minibatch)

    # unpack minibatch
    state_batch = torch.cat([d[0] for d in minibatch])
    # print([d[1] for d in minibatch])
    action_batch = torch.cat([d[1] for d in minibatch])
    reward_batch = torch.cat([d[2] for d in minibatch])
    new_state_batch = torch.cat([d[3] for d in minibatch])

    state_batch = state_batch.to(DEVICE)
    action_batch = action_batch.to(DEVICE).float()
    reward_batch = reward_batch.to(DEVICE)
    new_state_batch = new_state_batch.to(DEVICE)

    # # get output for the next state
    output_new_state_batch = policy(new_state_batch)

    # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
    # y_batch = torch.cat(
    #     [reward_batch[i] if minibatch[i][4]
    #      else reward_batch[i]+policy.gamma*torch.max(output_new_state_batch[i])
    #      for i in range(len(minibatch))
    #      ]
    # )
    listReward = []
    for i in range(len(minibatch)):
        if minibatch[i][4]:
            rwd = reward_batch[i]
            rwd = rwd.unsqueeze(-1)
        else:
            rwd = reward_batch[i]+policy.gamma * \
                torch.max(output_new_state_batch[i])
            rwd = rwd.unsqueeze(-1)
        listReward.append(rwd)

    y_batch = torch.cat(listReward)
    # print("LISTOUILLE : ", y_batch, y_batch.size())

    # extract Q-value
    # print(policy(state_batch).size())
    q_value = torch.sum(policy(state_batch).t() *
                        action_batch, dim=0)

    # PyTorch accumulates gradients by default,
    # so they need to be reset in each pass
    optimizer.zero_grad()

    # returns a new Tensor, detached from the current graph,
    # the result will never require gradient
    y_batch = y_batch.detach()

    # calculate loss
    # print("LKJDSLK :: ", q_value.size(), y_batch.size())
    loss = criterion(q_value, y_batch)

    # do backward pass
    loss.backward()
    optimizer.step()

    # set state to be state_1
    state = new_state

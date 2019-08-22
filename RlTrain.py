from environement import Environement
from RlModel import Policy
from torch.optim import Adam
from settings import LEARNING_RATE_RL, EPS_DECAY, EPS_END, EPS_START, DEVICE
import random
import numpy as np
import math
import torch
from torch import nn

environement = Environement()
environement.new_person()
policy = Policy()
policy = Policy.to(DEVICE)
print("Nombre de paramètres police: ",
      f"{sum([np.prod(p.size()) for p in policy.parameters()]):,}")

optimizer = Adam(policy.parameters(), lr=LEARNING_RATE_RL)

iteration = 0
done = False
criterion = nn.MSELoss()
while not done:
    print("k")
    state = environement.synth_im
    state = state.to(DEVICE)
    probas = policy(state)
    # Choose action
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * iteration / EPS_DECAY)
    iteration += 1
    if sample > eps_threshold:
        action_index = torch.argmax(probas)
    else:
        action_index = torch.randint(policy.action_space, dtype=torch.int)
    # Apply action
    new_state, reward, done = environement.step(action_index)
    policy.replay_memory.append((state, action_index, reward, new_state, done))
    minibatch = random.sample(policy.replay_memory,
                              min(len(policy.replay_memory), 8))

    # unpack minibatch
    state_batch = torch.cat(tuple(d[0] for d in minibatch))
    action_batch = torch.cat(tuple(d[1] for d in minibatch))
    reward_batch = torch.cat(tuple(d[2] for d in minibatch))
    state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

    state_batch = state_batch.to(DEVICE)
    action_batch = action_batch.to(DEVICE)
    reward_batch = reward_batch.to(DEVICE)
    state_1_batch = state_1_batch.to(DEVICE)

    # get output for the next state
    output_1_batch = policy(state_1_batch)

    # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
    y_batch = torch.cat(
        tuple(reward_batch[i] if minibatch[i][4]
              else reward_batch[i]+policy.gamma*torch.max(output_1_batch[i])
              for i in range(len(minibatch))
              )
    )

    # extract Q-value
    q_value = torch.sum(policy(state_batch) * action_batch, dim=1)

    # PyTorch accumulates gradients by default,
    # so they need to be reset in each pass
    optimizer.zero_grad()

    # returns a new Tensor, detached from the current graph,
    # the result will never require gradient
    y_batch = y_batch.detach()

    # calculate loss
    loss = criterion(q_value, y_batch)

    # do backward pass
    loss.backward()
    optimizer.step()

    # set state to be state_1
    state = new_state

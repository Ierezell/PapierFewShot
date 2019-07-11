from environement import Environement
from RlModel import Policy
from torch.optim import Adam
from settings import LEARNING_RATE_RL

environement = Environement()
policy = Policy()
optimizer = Adam(policy.parameters(), lr=LEARNING_RATE_RL)

iteration = 0
while iteration < environement.iterations:
    probas = policy(environement.synth_im)

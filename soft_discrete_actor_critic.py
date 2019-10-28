import copy
import logging
import os
import random
import sys
import time
from collections import deque, namedtuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal, normal
from torch.optim import Adam

from environementV2 import Environement
from losses import vgg_face_dag
from nn_builder.pytorch.NN import NN

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class lstm_Policy(nn.Module):
    def __init__(self):
        super(lstm_Policy, self).__init__()
        self.state_space = 1
        self.action_space = 68*2*224*224

        self.repres_image = vgg_face_dag(freeze=False)
        for name, param in self.repres_image.named_parameters():
            if "fc" in name:
                # print(name)
                param.requires_grad = False
            else:
                param.requires_grad = False

        grad_param_vgg = sum([np.prod(p.size()) if p.requires_grad else 0
                              for p in self.repres_image.parameters()])
        # print("Nombre de paramètres vggface: ", f"{grad_param_vgg:,}")

        # self.l1 = nn.Linear(self.state_space, 512, bias=False)
        self.rnn = nn.LSTM(input_size=self.state_space, hidden_size=512,
                           num_layers=2, bias=True, batch_first=True,
                           dropout=0.2, bidirectional=False)
        self.l2 = nn.Linear(512, self.action_space, bias=False)
        self.prev_img_repr = deque(maxlen=10)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU()
        self.gamma = 0.999
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


class fc_Policy(nn.Module):
    def __init__(self):
        super(fc_Policy, self).__init__()
        self.state_space = 1
        self.action_space = 68*2*224*224
        self.l2 = nn.Linear(self.state_space, self.action_space, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.softmax(self.l2(x))


def create_actor_distribution(action_types, actor_output, action_size):
    """
    Creates a distribution that the actor can then use to
    randomly draw actions
    """
    if action_types == "DISCRETE":
        assert actor_output.size(
        )[1] == action_size, "Actor output the wrong size"
        # this creates a distribution to sample from
        action_distribution = Categorical(actor_output)
    else:
        assert actor_output.size()[1] == action_size * \
            2, "Actor output the wrong size"
        means = actor_output[:, :action_size].squeeze(0)
        stds = actor_output[:,  action_size:].squeeze(0)
        if len(means.shape) == 2:
            means = means.squeeze(-1)
        if len(stds.shape) == 2:
            stds = stds.squeeze(-1)
        if len(stds.shape) > 1 or len(means.shape) > 1:
            raise ValueError(
                f"Wrong mean and std shapes - {stds.shape} -- { means.shape}")
        action_distribution = normal.Normal(means.squeeze(0), torch.abs(stds))
    return action_distribution


class OU_Noise(object):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * \
            np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state


class Replay_Buffer(object):
    """
    Replay buffer to store past experiences that the agent can then use
    for training data
    """

    def __init__(self, buffer_size, batch_size, seed):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward",
                                     "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(state, action, reward,
                                           next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(
                states, actions, rewards, next_states, dones)
            self.memory.append(experience)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            (states, actions, rewards,
             next_states, dones) = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences

    def separate_out_data_types(self, experiences):
        """
        Puts the sampled experience into the correct format for a
        PyTorch neural network
        """
        states = torch.from_numpy(np.vstack(
            [e.state
             for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack(
            [e.action
             for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward
             for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state
             for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack(
            [int(e.done)
             for e in experiences if e is not None])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class Config(object):
    """Object to hold the config requirements for an agent/game"""

    def __init__(self):
        self.seed = None
        self.environment = None
        self.requirements_to_solve_game = None
        self.num_episodes_to_run = None
        self.file_to_save_data_results = None
        self.file_to_save_results_graph = None
        self.runs_per_agent = None
        self.visualise_overall_results = None
        self.visualise_individual_results = None
        self.hyperparameters = None
        self.use_GPU = None
        self.overwrite_existing_results_file = None
        self.save_model = False
        self.standard_deviation_results = 1.0
        self.randomise_random_seed = True
        self.show_solution_score = False
        self.debug_mode = False


class Base_Agent(object):

    def __init__(self, config):
        self.logger = self.setup_logger()
        self.debug_mode = config.debug_mode
        # if self.debug_mode: self.tensorboard = SummaryWriter()
        self.config = config
        self.set_random_seeds(config.seed)
        self.environment = config.environment
        self.environment_title = self.get_environment_title()
        self.action_types = "DISCRETE" if self.environment.action_space.dtype == np.int64 else "CONTINUOUS"
        self.action_size = int(self.get_action_size())
        self.config.action_size = self.action_size

        self.lowest_possible_episode_score = self.get_lowest_possible_episode_score()

        self.state_size = int(self.get_state_size())
        self.hyperparameters = config.hyperparameters
        self.average_score_required_to_win = self.get_score_required_to_win()
        self.rolling_score_window = self.get_trials()
        # self.max_steps_per_episode = self.environment.spec.max_episode_steps
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")
        self.max_episode_score_seen = float("-inf")
        self.episode_number = 0
        self.device = "cuda:0" if config.use_GPU else "cpu"
        self.visualise_results_boolean = config.visualise_individual_results
        self.global_step_number = 0
        self.turn_off_exploration = False
        # stops it from printing an unnecessary warning
        gym.logger.set_level(40)
        self.log_game_info()

    def step(self):
        """Takes a step in the game. This method must be overriden by any agent"""
        raise ValueError("Step needs to be implemented by the agent")

    def get_environment_title(self):
        """Extracts name of environment from it"""
        try:
            name = self.environment.unwrapped.id
        except AttributeError:
            try:
                if str(self.environment.unwrapped)[1:11] == "FetchReach":
                    return "FetchReach"
                elif str(self.environment.unwrapped)[1:8] == "AntMaze":
                    return "AntMaze"
                elif str(self.environment.unwrapped)[1:7] == "Hopper":
                    return "Hopper"
                elif str(self.environment.unwrapped)[1:9] == "Walker2d":
                    return "Walker2d"
                else:
                    name = self.environment.spec.id.split("-")[0]
            except AttributeError:
                name = str(self.environment.env)
                if name[0:10] == "TimeLimit<":
                    name = name[10:]
                name = name.split(" ")[0]
                if name[0] == "<":
                    name = name[1:]
                if name[-3:] == "Env":
                    name = name[:-3]
        return name

    def get_lowest_possible_episode_score(self):
        """Returns the lowest possible episode score you can get in an environment"""
        if self.environment_title == "Taxi":
            return -800
        return None

    def get_action_size(self):
        """Gets the action_size for the gym env into the correct shape for a neural network"""
        if "overwrite_action_size" in self.config.__dict__:
            return self.config.overwrite_action_size
        if "action_size" in self.environment.__dict__:
            return self.environment.action_size
        if self.action_types == "DISCRETE":
            return self.environment.action_space.n
        else:
            return self.environment.action_space.shape[0]

    def get_state_size(self):
        """Gets the state_size for the gym env into the correct shape for a neural network"""
        random_state = self.environment.reset()
        if isinstance(random_state, dict):
            state_size = random_state["observation"].shape[0] + \
                random_state["desired_goal"].shape[0]
            return state_size
        else:
            return random_state.size

    def get_score_required_to_win(self):
        """Gets average score required to win game"""
        print("TITLE ", self.environment_title)
        if self.environment_title == "FetchReach":
            return -5
        if self.environment_title in ["AntMaze", "Hopper",
                                      "Walker2d", "FaceEnv"]:
            print(
                "Score required to win set to infinity therefore no learning rate annealing will happen")
            return float("inf")
        try:
            return self.environment.unwrapped.reward_threshold
        except AttributeError:
            try:
                return self.environment.spec.reward_threshold
            except AttributeError:
                return self.environment.unwrapped.spec.reward_threshold

    def get_trials(self):
        """Gets the number of trials to average a score over"""
        if self.environment_title in ["AntMaze", "FetchReach", "FaceEnv",
                                      "Hopper", "Walker2d", "CartPole"]:
            return 100
        try:
            return self.environment.unwrapped.trials
        except AttributeError:
            return self.environment.spec.trials

    def setup_logger(self):
        """Sets up the logger"""
        filename = "Training.log"
        try:
            if os.path.isfile(filename):
                os.remove(filename)
        except:
            pass

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # create a file handler
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        # create a logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(handler)
        return logger

    def log_game_info(self):
        """Logs info relating to the game"""
        for ix, param in enumerate([self.environment_title, self.action_types, self.action_size, self.lowest_possible_episode_score,
                                    self.state_size, self.hyperparameters, self.average_score_required_to_win, self.rolling_score_window,
                                    self.device]):
            self.logger.info("{} -- {}".format(ix, param))

    def set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        # tf.set_random_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)
        if hasattr(gym.spaces, 'prng'):
            gym.spaces.prng.seed(random_seed)

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        self.environment.seed(self.config.seed)
        self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        if "exploration_strategy" in self.__dict__.keys():
            self.exploration_strategy.reset()
        self.logger.info(
            "Reseting game -- New start state {}".format(self.state))

    def track_episodes_data(self):
        """Saves the data from the recent episodes"""
        self.episode_states.append(self.state)
        self.episode_actions.append(self.action)
        self.episode_rewards.append(self.reward)
        self.episode_next_states.append(self.next_state)
        self.episode_dones.append(self.done)

    def run_n_episodes(self, num_episodes=None, show_whether_achieved_goal=True, save_and_print_results=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)"""
        if num_episodes is None:
            num_episodes = self.config.num_episodes_to_run
        start = time.time()
        while self.episode_number < num_episodes:
            self.reset_game()
            self.step()
            if save_and_print_results:
                self.save_and_print_result()
        time_taken = time.time() - start
        if show_whether_achieved_goal:
            self.show_whether_achieved_goal()
        if self.config.save_model:
            self.locally_save_policy()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, _ = self.environment.step(
            action)
        self.total_episode_score_so_far += self.reward
        if self.hyperparameters["clip_rewards"]:
            self.reward = max(min(self.reward, 1.0), -1.0)

    def save_and_print_result(self):
        """Saves and prints results of the game"""
        self.save_result()
        self.print_rolling_result()

    def save_result(self):
        """Saves the result of an episode of the game"""
        self.game_full_episode_scores.append(self.total_episode_score_so_far)
        self.rolling_results.append(
            np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()

    def save_max_result_seen(self):
        """Updates the best episode result seen so far"""
        if self.game_full_episode_scores[-1] > self.max_episode_score_seen:
            self.max_episode_score_seen = self.game_full_episode_scores[-1]

        if self.rolling_results[-1] > self.max_rolling_score_seen:
            if len(self.rolling_results) > self.rolling_score_window:
                self.max_rolling_score_seen = self.rolling_results[-1]

    def print_rolling_result(self):
        """Prints out the latest episode results"""
        text = """"\r Episode {0}, Score: {3: .2f}, Max score seen: {4: .2f}, Rolling score: {1: .2f}, Max rolling score seen: {2: .2f}"""
        sys.stdout.write(text.format(len(self.game_full_episode_scores), self.rolling_results[-1], self.max_rolling_score_seen,
                                     self.game_full_episode_scores[-1], self.max_episode_score_seen))
        sys.stdout.flush()

    def show_whether_achieved_goal(self):
        """Prints out whether the agent achieved the environment target goal"""
        index_achieved_goal = self.achieved_required_score_at_index()
        print(" ")
        if index_achieved_goal == -1:  # this means agent never achieved goal
            print("\033[91m" + "\033[1m" +
                  "{} did not achieve required score \n".format(self.agent_name) +
                  "\033[0m" + "\033[0m")
        else:
            print("\033[92m" + "\033[1m" +
                  "{} achieved required score at episode {} \n".format(self.agent_name, index_achieved_goal) +
                  "\033[0m" + "\033[0m")

    def achieved_required_score_at_index(self):
        """Returns the episode at which agent achieved goal or -1 if it never achieved it"""
        for ix, score in enumerate(self.rolling_results):
            if score > self.average_score_required_to_win:
                return ix
        return -1

    def update_learning_rate(self, starting_lr,  optimizer):
        """Lowers the learning rate according to how close we are to the solution"""
        if len(self.rolling_results) > 0:
            last_rolling_score = self.rolling_results[-1]
            if last_rolling_score > 0.75 * self.average_score_required_to_win:
                new_lr = starting_lr / 100.0
            elif last_rolling_score > 0.6 * self.average_score_required_to_win:
                new_lr = starting_lr / 20.0
            elif last_rolling_score > 0.5 * self.average_score_required_to_win:
                new_lr = starting_lr / 10.0
            elif last_rolling_score > 0.25 * self.average_score_required_to_win:
                new_lr = starting_lr / 2.0
            else:
                new_lr = starting_lr
            for g in optimizer.param_groups:
                g['lr'] = new_lr
        if random.random() < 0.001:
            self.logger.info("Learning rate {}".format(new_lr))

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > self.hyperparameters["batch_size"]

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None:
            memory = self.memory
        if experience is None:
            experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list):
            network = [network]
        optimizer.zero_grad()  # reset gradients to 0
        # this calculates the gradients
        loss.backward(retain_graph=retain_graph)
        self.logger.info("Loss -- {}".format(loss.item()))
        if self.debug_mode:
            self.log_gradient_and_weight_information(network, optimizer)
        if clipping_norm is not None:
            for net in network:
                # clip gradients to help stabilise training
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm)
        optimizer.step()  # this applies the gradients

    def log_gradient_and_weight_information(self, network, optimizer):

        # log weight information
        total_norm = 0
        for name, param in network.named_parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.logger.info("Gradient Norm {}".format(total_norm))

        for g in optimizer.param_groups:
            learning_rate = g['lr']
            break
        self.logger.info("Learning Rate {}".format(learning_rate))

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

    def create_NN(self, input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
        """Creates a neural network for the agents to use"""
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        if key_to_use:
            hyperparameters = hyperparameters[key_to_use]
        if override_seed:
            seed = override_seed
        else:
            seed = self.config.seed

        default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
                                          "initialiser": "default", "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [], "y_range": ()}

        for key in default_hyperparameter_choices:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameter_choices[key]

        return NN(input_dim=input_dim, layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
                  output_activation=hyperparameters["final_layer_activation"],
                  batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                  hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                  columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                  embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
                  random_seed=seed).to(self.device)

    def turn_on_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning on epsilon greedy exploration")
        self.turn_off_exploration = False

    def turn_off_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning off epsilon greedy exploration")
        self.turn_off_exploration = True

    def freeze_all_but_output_layers(self, network):
        """Freezes all layers except the output layer of a network"""
        print("Freezing hidden layers")
        for param in network.named_parameters():
            param_name = param[0]
            assert "hidden" in param_name or "output" in param_name or "embedding" in param_name, "Name {} of network layers not understood".format(
                param_name)
            if "output" not in param_name:
                param[1].requires_grad = False

    def unfreeze_all_layers(self, network):
        """Unfreezes all layers of a network"""
        print("Unfreezing all layers")
        for param in network.parameters():
            param.requires_grad = True

    @staticmethod
    def move_gradients_one_model_to_another(from_model, to_model, set_from_gradients_to_zero=False):
        """Copies gradients from from_model to to_model"""
        for from_model, to_model in zip(from_model.parameters(), to_model.parameters()):
            to_model._grad = from_model.grad.clone()
            if set_from_gradients_to_zero:
                from_model._grad = None

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class SAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "CONTINUOUS", "Action types must be continuous. Use SAC Discrete instead for discrete actions"
        assert self.config.hyperparameters["Actor"][
            "final_layer_activation"] != "Softmax", "Final actor layer must not be softmax"
        self.hyperparameters = config.hyperparameters
        self.critic_local = self.create_NN(
            input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                             key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                            key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                              key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)
        self.actor_local = self.create_NN(
            input_dim=self.state_size, output_dim=self.action_size * 2, key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters[
            "automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(
                self.device)).item()  # heuristic value from the paper
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam(
                [self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

    def save_result(self):
        """Saves the result of an episode of the game. Overriding the method in Base Agent that does this because we only
        want to keep track of the results during the evaluation episodes"""
        if self.episode_number == 1 or not self.do_evaluation_iterations:
            self.game_full_episode_scores.extend(
                [self.total_episode_score_so_far])
            self.rolling_results.append(
                np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
            self.save_max_result_seen()

        elif (self.episode_number - 1) % TRAINING_EPISODES_PER_EVAL_EPISODE == 0:
            self.game_full_episode_scores.extend(
                [self.total_episode_score_so_far for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.rolling_results.extend([np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:])
                                         for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.save_max_result_seen()

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)
        if self.add_extra_noise:
            self.noise.reset()

    def step(self):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
        eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
        self.episode_step_number_val = 0
        while not self.done:
            self.episode_step_number_val += 1
            self.action = self.pick_action(eval_ep)
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
            mask = False if self.episode_step_number_val >= self.environment._max_episode_steps else self.done
            if not eval_ep:
                self.save_experience(experience=(
                    self.state, self.action, self.reward, self.next_state, mask))
            self.state = self.next_state
            self.global_step_number += 1
        print(self.total_episode_score_so_far)
        if eval_ep:
            self.print_summary_of_latest_evaluation_episode()
        self.episode_number += 1

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        if state is None:
            state = self.state
        if eval_ep:
            action = self.actor_pick_action(state=state, eval=True)
        elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            action = self.environment.action_space.sample()
            print("Picking random action ", action)
        else:
            action = self.actor_pick_action(state=state)
        if self.add_extra_noise:
            action += self.noise.sample()
        return action

    def actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None:
            state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if eval == False:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        actor_output = self.actor_local(state)
        mean, log_std = (actor_output[:, :self.action_size],
                         actor_output[:, self.action_size:])
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
            self.enough_experiences_to_learn_from(
        ) and self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        qf1_loss, qf2_loss = self.calculate_critic_losses(
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        if self.automatic_entropy_tuning:
            alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else:
            alpha_loss = None
        self.update_all_parameters(qf1_loss, qf2_loss, policy_loss, alpha_loss)

    def sample_experiences(self):
        return self.memory.sample()

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(
                next_state_batch)
            qf1_next_target = self.critic_target(
                torch.cat((next_state_batch, next_state_action), 1))
            qf2_next_target = self.critic_target_2(
                torch.cat((next_state_batch, next_state_action), 1))
            min_qf_next_target = torch.min(
                qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + \
                (1.0 - mask_batch) * \
                self.hyperparameters["discount_rate"] * (min_qf_next_target)
        qf1 = self.critic_local(torch.cat((state_batch, action_batch), 1))
        qf2 = self.critic_local_2(torch.cat((state_batch, action_batch), 1))
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(torch.cat((state_batch, action), 1))
        qf2_pi = self.critic_local_2(torch.cat((state_batch, action), 1))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi +
                                         self.target_entropy).detach()).mean()
        return alpha_loss

    def update_all_parameters(self, critic_loss_1, critic_loss_2, actor_loss, alpha_loss):
        """Updates the parameters for the actor, both critics and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])
        if alpha_loss is not None:
            self.take_optimisation_step(
                self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print("Episode score {} ".format(self.total_episode_score_so_far))
        print("----------------------------")


class SAC_Discrete(SAC):
    """The Soft Actor Critic for discrete actions. It inherits from SAC
    for continuous actions and only changes a few
    methods."""
    agent_name = "SAC"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "DISCRETE", "Action types must be discrete. Use SAC instead for continuous actions"
        self.hyperparameters = config.hyperparameters
        self.critic_local = fc_Policy().to("cuda")
        self.critic_local_2 = fc_Policy().to("cuda")
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=0.001, eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=0.001, eps=1e-4)
        self.critic_target = fc_Policy().to("cuda")
        self.critic_target_2 = fc_Policy().to("cuda")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(1000, 2, self.config.seed)

        self.actor_local = fc_Policy().to("cuda")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=0.001, eps=1e-4)
        self.automatic_entropy_tuning = True
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam(
                [self.log_alpha], lr=0.001, eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]
        self.add_extra_noise = False
        self.do_evaluation_iterations = True

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the probability of the action,
         the log probability of the action, and
        the argmax action"""
        action_probabilities = self.actor_local(state)
        max_probability_action = torch.argmax(
            action_probabilities).unsqueeze(0)
        action_distribution = create_actor_distribution(
            self.action_types, action_probabilities, self.action_size)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(
                next_state_batch)
            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * \
                (torch.min(qf1_next_target, qf2_next_target) -
                 self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.mean(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + \
                (1.0 - mask_batch) * \
                self.hyperparameters["discount_rate"] * (min_qf_next_target)

        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, (action_probabilities,
                 log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = action_probabilities * inside_term
        policy_loss = policy_loss.mean()
        log_action_probabilities = torch.sum(
            log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities


config = Config()
config.seed = 666
config.environment = Environement()
config.num_episodes_to_run = 100
config.file_to_save_data_results = "./testRlSave"
config.file_to_save_results_graph = "./testRlSaveGraph"
config.runs_per_agent = 10
config.visualise_overall_results = True
config.visualise_individual_results = True
config.use_GPU = True
config.overwrite_existing_results_file = True
config.save_model = True
config.standard_deviation_results = 1.0
config.randomise_random_seed = True
config.show_solution_score = False
config.debug_mode = True
config.hyperparameters = {
    "Actor": {
                "learning_rate": 0.003,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": None,
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

            "Critic": {
                "learning_rate": 0.02,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 1000000,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

        "min_steps_before_learning": 1000, #for SAC only
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 20,
        "learning_updates_per_learning_session": 10,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": True,
        "do_evaluation_iterations": True,
        "clip_rewards": False

}
sac = SAC(config)
sac.run_n_episodes(100)

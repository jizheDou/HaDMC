import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agent import Agent
from noise import OrnsteinUhlenbeckActionNoise


class QActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size):
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size
        self.layers.append(nn.Linear(inputSize, 256))
        self.layers.append(nn.Linear(256, 256))
        self.layers.append(nn.Linear(256, self.action_size))

    def forward(self, state, action_parameters):
        x = torch.cat((state, action_parameters), dim=1)
        x = F.relu(self.layers[0](x))
        x = F.relu(self.layers[1](x))
        Q = self.layers[-1](x)

        return Q


class ParamActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size):
        super(ParamActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        # self.squashing_function = squashing_function

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size
        self.layers.append(nn.Linear(inputSize, 256))
        self.layers.append(nn.Linear(256, 256))

        self.action_parameters_output_layer = nn.Linear(256, self.action_parameter_size)
        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)

        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)
        # fix passthrough layer to avoid instability, rest of network can compensate
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        x = state

        x = F.relu(self.layers[0](x))
        x = F.relu(self.layers[1](x))
        action_params = self.action_parameters_output_layer(x)
        action_params += self.action_parameters_passthrough_layer(state)
        # if self.squashing_function:
        #     assert False  # scaling not implemented yet
        #     action_params = action_params.tanh()
        #     action_params = action_params * self.action_param_lim
        return action_params


class PDQNAgent(Agent):
    """
    DDPG actor-critic agent for parameterised action spaces
    [Hausknecht and Stone 2016]
    """

    NAME = "P-DQN Agent"

    def __init__(self,
                 observation_space,
                 action_space,
                 actor_class=QActor,
                 actor_param_class=ParamActor,
                 epsilon_initial=1.0,
                 epsilon_final=0.05,
                 epsilon_steps=10000,
                 batch_size=64,
                 gamma=0.99,
                 tau_actor=0.01,  # Polyak averaging factor for copying target weights
                 tau_actor_param=0.001,
                 replay_memory_size=1000000,
                 learning_rate_actor=0.0001,
                 learning_rate_actor_param=0.00001,
                 initial_memory_threshold=0,
                 use_ornstein_noise=False,
                 # if false, uses epsilon-greedy with uniform-random action-parameter exploration
                 loss_func=F.mse_loss,  # F.mse_loss
                 clip_grad=10,
                 inverting_gradients=False,
                 zero_index_gradients=False,
                 indexed=False,
                 weighted=False,
                 average=False,
                 random_weighted=False,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None):

        super(PDQNAgent, self).__init__(observation_space, action_space)

        super(PDQNAgent, self).__init__(observation_space, action_space)
        self.device = torch.device(device)
        self.num_actions = action_space
        self.action_parameter_sizes = np.array([1, 1])
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)
        self.use_ornstein_noise = use_ornstein_noise
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, random_machine=np.random, mu=0.,
                                                  theta=0.15, sigma=0.0001)  # , theta=0.01, sigma=0.01)
        self.actor = actor_class(self.observation_space, self.num_actions, self.action_parameter_size).to(device)
        self.actor_param = actor_param_class(self.observation_space, self.num_actions, self.action_parameter_size).to(
            device)
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max - self.action_min).detach()
        self.action_parameter_max_numpy = np.array([1, 1])
        self.action_parameter_min_numpy = np.array([-1, -1])
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            all_action_parameters = self.actor_param.forward(state)
            rnd = np.random.uniform()
            if rnd < 1.0:
                action = np.random.choice(self.num_actions)
                if not self.use_ornstein_noise:
                    all_action_parameters = torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                                               self.action_parameter_max_numpy))
            else:
                Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                action = np.argmax(Q_a)  # 返回最大离散动作的索引

            # add noise only to parameters of chosen action
            all_action_parameters = all_action_parameters.cpu().data.numpy()
            action_tmp1 = action
            action = int(action / (self.num_actions / 2))
            offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
            if self.use_ornstein_noise and self.noise is not None:
                all_action_parameters[offset:offset + self.action_parameter_sizes[action]] += self.noise.sample()[
                                                                                              offset:offset +
                                                                                                     self.action_parameter_sizes[
                                                                                                         action]]

            action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]

        return action_tmp1, action_parameters, all_action_parameters

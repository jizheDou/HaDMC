import numpy as np
import torch
np.random.seed(0)
torch.manual_seed(0)
dtype = torch.FloatTensor

def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

    Advantage: Less memory requirement O(M*d + N*d + M*N) instead of O(N*M*d)
    Computationally more expensive? Maybe, Not sure.
    adapted from: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    '''
    # print("x",x)
    # print("y",y)

    x_norm = (x ** 2).sum(1).view(-1, 1)  # sum(1)将一个矩阵的每一行向量相加
    y_norm = (y ** 2).sum(1).view(1, -1)
    # print("x_norm",x_norm)
    # print("y_norm",y_norm)
    y_t = torch.transpose(y, 0, 1)  # 交换一个tensor的两个维度
    # a^2 + b^2 - 2ab
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)  # torch.mm 矩阵a和b矩阵相乘
    # dist[dist != dist] = 0 # replace nan values with 0
    # print("dist",dist)
    return dist

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X

def decode_discrete_action(discrete_action, action_dims):
    """
    Change the actions format to select the stopping time, decode actions.
    action = 0, uav 0, charger 0
    action = 1, uav 0, charger 1
    action = 2, uav 1, charger 0
    action = 3, uav 1, charger 1
    :param discrete_action:
    :return:
    """
    action_tmp = list()
    action_1 = int(discrete_action / (action_dims / 2))
    action_2 = discrete_action % int((action_dims / 2))
    action_tmp.append(action_1)
    action_tmp.append(action_2)
    return action_tmp


class ReplayBuffer(object):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim, all_parameter_action_dim, discrete_emb_dim,
                 parameter_emb_dim,
                 max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.discrete_action = np.zeros((max_size, discrete_action_dim))
        self.parameter_action = np.zeros((max_size, parameter_action_dim))
        self.all_parameter_action = np.zeros((max_size, all_parameter_action_dim))

        self.discrete_emb = np.zeros((max_size, discrete_emb_dim))
        self.parameter_emb = np.zeros((max_size, parameter_emb_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.state_next_state = np.zeros((max_size, state_dim))

        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, discrete_action, parameter_action, all_parameter_action, discrete_emb, parameter_emb,
            next_state, state_next_state, reward, done):
        self.state[self.ptr] = state
        self.discrete_action[self.ptr] = discrete_action
        self.parameter_action[self.ptr] = parameter_action
        self.all_parameter_action[self.ptr] = all_parameter_action
        self.discrete_emb[self.ptr] = discrete_emb
        self.parameter_emb[self.ptr] = parameter_emb
        self.next_state[self.ptr] = next_state
        self.state_next_state[self.ptr] = state_next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, training_type):
        ind = np.random.randint(0, self.size, size=batch_size)
        if training_type == 0:
            return (
                torch.from_numpy(self.state[ind]).float().to(self.device),
                torch.from_numpy(self.discrete_action[ind]).float().to(self.device),
                torch.from_numpy(self.parameter_action[ind]).float().to(self.device),
            )
        return (
            torch.from_numpy(self.state[ind]).float().to(self.device),
            torch.from_numpy(self.discrete_emb[ind]).float().to(self.device),
            torch.from_numpy(self.parameter_emb[ind]).float().to(self.device),
            torch.from_numpy(self.next_state[ind]).float().to(self.device),
            torch.from_numpy(self.reward[ind]).float().to(self.device),
            torch.from_numpy(self.not_done[ind]).float().to(self.device)
        )

import pandas as pd
import numpy as np
import simpy
import torch
from HaDMC.utils import decode_discrete_action
from world.GreedyObservation import GreedyObservation
from collections import Counter


def count_boundary(c_rate):
    median = (c_rate[0] - c_rate[1]) / 2
    offset = c_rate[0] - 1 * median
    return median, offset


def true_parameter_action(parameter_action, c_rate):
    parameter_action_ = parameter_action
    for i in range(len(parameter_action)):
        median, offset = count_boundary(c_rate[i])
        parameter_action_[i] = parameter_action_[i] * median + offset
    return parameter_action_


def pad_action(act, act_param, decoupling_dim):
    drone_discrete_action = int(act / decoupling_dim)
    charger_discrete_action = int(act % decoupling_dim)
    if len(act_param) == 1:
        parameter_action = act_param
    else:
        parameter_action = act_param[drone_discrete_action]
    parameter_action_tmp = parameter_action
    if drone_discrete_action == 0:
        parameter_action = 4 * parameter_action + 4
    else:
        parameter_action = 5 * parameter_action + 5
    if isinstance(parameter_action, np.ndarray):
        parameter_action = parameter_action[0]
    if len(act_param) != 1:
        return [drone_discrete_action, parameter_action, charger_discrete_action], parameter_action_tmp
    else:
        return [drone_discrete_action, parameter_action, charger_discrete_action]


class Evaluation:

    def __init__(self, env, policy, action_rep, device):
        self.env = env
        self.policy = policy
        self.action_rep = action_rep
        self.action_list = list()
        self.charging_position_list = list()
        self.greedy_action_list = list()
        self.device = device
        self.greedy_results = None
        self.dcarl_results = None
        self.td3_results = None

    def evaluate(self, episodes=100):
        state = self.env.reset()
        self.action_list = list()
        self.charging_position_list = list()
        self.greedy_action_list = list()
        greedy = GreedyObservation(self.env)
        greedy_objective, self.greedy_action_list = greedy.run()
        if self.greedy_results is None:
            self.greedy_results = np.array(greedy.print_time())
        else:
            self.greedy_results = np.vstack((self.greedy_results, np.array(greedy.print_time())))
        terminal = False
        count = 0
        total_reward = 0.
        while not terminal:
            count += 1
            state = torch.from_numpy(state.reshape(1, -1)).float()
            state = state.to(self.device)
            all_discrete_action, all_parameter_action = self.policy(state)
            discrete_emb, parameter_emb = (all_discrete_action, all_parameter_action)
            true_parameter_emb = parameter_emb
            state = state.cpu()
            true_parameter_emb = true_parameter_emb.cpu()
            discrete_emb = discrete_emb.cpu()
            all_parameter_action = self.action_rep.select_parameter_action(
                true_parameter_emb)
            discrete_action_embedding = discrete_emb
            discrete_action_embedding = discrete_action_embedding.float().reshape(1, -1)
            discrete_action = self.action_rep.select_discrete_action(
                discrete_action_embedding)
            action_tmp = decode_discrete_action(discrete_action, 2 * len(self.env.scenario.charging_tables))
            action = list()
            if torch.is_tensor(all_parameter_action):
                all_parameter_action = all_parameter_action.cpu()
            if action_tmp[0] == 1:
                parameter_action = 4 * (all_parameter_action + 1) / 2 + 4
            else:
                parameter_action = 5 * (all_parameter_action + 1) / 2 + 5
            action.append(action_tmp[0])
            action.append(parameter_action[0])
            action.append(action_tmp[1])
            action_tuple = tuple(action)
            state, reward, terminal = self.env.step(action)
            total_reward += reward
            self.charging_position_list.append(self.env.scenario.charger.position)
            self.action_list.append(action)  # parameter action changes
        self.action_list[-1][2] = -1
        dcarl_objective = self.env.task_finished_ratio() / self.env.sum_time
        dcarl_result = np.array((float(self.env.scenario.sum_observation_time),
                                 float(self.env.scenario.sum_charging_time), self.env.scenario.sum_waiting_time,
                                 self.env.scenario.sum_flying_time, self.env.sum_time, dcarl_objective))
        if self.dcarl_results is None:
            self.dcarl_results = dcarl_result
        else:
            self.dcarl_results = np.vstack((self.dcarl_results, dcarl_result))
        return greedy_objective, dcarl_objective


    def random_simulation(self):
        n = self.action_rep.reduced_action_dim
        simulated_action = (torch.rand(100000, n) - 0.5) * 2
        discrete_action = self.action_rep.select_discrete_action(simulated_action)
        count = Counter(discrete_action)
        x = []
        y = []
        for i in range(0, 2 * len(self.env.scenario.charging_tables)):
            y.append(count[i])
            x.append(i)
        var = np.var(y)
        return var

    def random_simulation_continuous(self):
        n = self.action_rep.reduce_parameter_action_dim
        true_parameter_emb = (torch.rand(100000, n) - 0.5) * 2
        all_parameter_action = self.action_rep.select_parameter_action(true_parameter_emb)
        all_parameter_action = pd.Series(all_parameter_action)
        range_list = np.arange(-1, 1 + 1 / 16, 1 / 16)
        range_list = range_list.tolist()
        se1 = pd.cut(all_parameter_action, range_list)
        counts = se1.value_counts()
        name = counts.axes
        count_list = []
        i = 0
        name_list = []
        for count in counts:
            name_list.append(name[0][i])
            i += 1
            count_list.append(count)
        var = np.var(count_list)
        return var


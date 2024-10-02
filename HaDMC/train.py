# ！/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/7/11 9:06
# @Author  : doujizhe
# @File    : Run
# @Software: PyCharm
from random import choice

import numpy as np
import argparse
import os
import sys
import simpy
import collections
import utils
import TD3
import copy

from Evaluation import Evaluation
from world.GreedyObservation import GreedyObservation

sys.path.append('../')
from scenario.scenario import CircleScenario
from pdqn import PDQNAgent
import AAE
import time
import torch
from world.world import TestWorld
from Logger import Logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_action(act, all_action_parameters, action_dims):
    """
    Change the actions format to select the stopping time, decode actions.
    action = 0, uav 0, charger 0
    action = 1, uav 0, charger 1
    action = 2, uav 1, charger 0
    action = 3, uav 1, charger 1
    :param action:
    :return:
    """
    action_tmp = list()
    action_1 = int(act / (action_dims / 2))
    action_2 = act % int((action_dims / 2))
    action_tmp.append(action_1)
    if action_tmp[0] == 1:
        parameter_action_tmp = 4 * (all_action_parameters[action_1] + 1) / 2 + 4
    else:
        parameter_action_tmp = 5 * (all_action_parameters[action_1] + 1) / 2 + 5
    action_tmp.append(parameter_action_tmp)
    action_tmp.append(action_2)
    return action_tmp


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def evaluate(env, policy, action_rep, log, episodes=100):
    returns = []
    epioside_steps = []
    epioside_times = []
    epioside_sum_times = []
    task_finished_ratios = []
    objectives = []
    greedy_objectives = []
    finished_list = []

    for _ in range(episodes):
        state = env.reset()
        greedy_objective, action_list = GreedyObservation(env).run()
        greedy_objectives.append(greedy_objective)
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            discrete_emb, parameter_emb = policy.select_action(state)
            true_parameter_emb = parameter_emb
            all_parameter_action = action_rep.select_parameter_action(
                true_parameter_emb)
            discrete_action_embedding = copy.deepcopy(discrete_emb)
            discrete_action_embedding = torch.from_numpy(
                discrete_action_embedding).float().reshape(1, -1)
            discrete_action = action_rep.select_discrete_action(
                discrete_action_embedding)
            action_tmp = utils.decode_discrete_action(discrete_action, 2 * len(env.scenario.charging_tables))
            action = list()
            action.append(action_tmp[0])
            if action_tmp[0] == 1:
                parameter_action = 4 * (all_parameter_action + 1) / 2 + 4
            else:
                parameter_action = 5 * (all_parameter_action + 1) / 2 + 5
            action.append(parameter_action[0])
            action.append(action_tmp[1])
            state, reward, terminal = env.step(action)
            total_reward += reward
        epioside_steps.append(t)
        finished_ratio = env.task_finished_ratio()
        task_finished_ratios.append(finished_ratio)
        epioside_times.append(env.scenario.uav.count_time)
        finished_ratio = finished_ratio / (env.sum_time + 1)
        objectives.append(finished_ratio)
        epioside_sum_times.append(env.sum_time)
        returns.append(total_reward)
        finished_list.append(env.finished)
    log.logger.info(str(np.array(returns[-episodes:]).mean()) + " " +
                    str(np.array(returns[-episodes:]).min()) + " " +
                    str(np.array(returns[-episodes:]).max()) + " " +
                    str(np.array(epioside_steps[-episodes:]).mean()) + " " +
                    str(np.array(epioside_sum_times[-episodes:]).mean()) + " " +
                    str(np.array(objectives[-episodes:]).mean()) + " " +
                    str(np.array(greedy_objectives[-episodes:]).mean()) + " " +
                    str(np.array(finished_list[-episodes:]).mean())
                    )
    return np.array(
        np.array(objectives[-episodes:]).mean()), np.array(finished_list[-100:]).mean()


def run(args, discrete_emb, parameter_emb):
    time_env = simpy.Environment()
    s1 = CircleScenario('s1', env=time_env)
    env = TestWorld(s1)
    env.reset()
    state_dim = env.scenario.observation_space_dimension
    action_dims = 2 * len(env.scenario.charging_tables)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Set environment parameters
    discrete_action_dim = action_dims
    parameter_action_dim = 2
    print("test", action_dims)
    discrete_emb_dim = discrete_emb
    parameter_emb_dim = parameter_emb
    max_action = 1.
    print("state_dim", state_dim)
    print("state_dim", state_dim)
    print("discrete_action_dim", discrete_action_dim)
    print("parameter_action_dim", parameter_action_dim)
    kwargs = {"state_dim": state_dim, "discrete_action_dim": discrete_emb_dim,
              "parameter_action_dim": parameter_emb_dim, "max_action": max_action, "discount": args.discount,
              "tau": args.tau, "policy_noise": args.policy_noise * max_action,
              "noise_clip": args.noise_clip * max_action, "policy_freq": args.policy_freq}
    policy = TD3.TD3(**kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data log
    time_now = time.strftime("%Y%m%d", time.localtime())
    root = os.getcwd()
    print(root)
    path = root + "\\results" + '\\' + str(env.type) + "_" + str(
        len(env.scenario.interesting_points) - 1) + '\\' + time_now + '\\' + \
           "actor_loss_" + str(policy.lr_actor)
    if not os.path.exists(path):
        os.makedirs(path)
    time_str = time.strftime('%Y-%m-%d %H %M %S', time.localtime())
    txt_path = str(discrete_emb_dim) + "_" + str(parameter_emb_dim) + "_" + time_str + ".txt"
    txt_path = path + '\\' + txt_path
    print(txt_path)
    if os.path.exists(txt_path):
        os.remove(txt_path)
    log = Logger('info', txt_path)

    # embedding初始部分 changed
    action_rep = AAE.Action_representation(
        state_dim=state_dim,
        action_dim=discrete_action_dim,
        parameter_action_dim=1,
        reduced_action_dim=discrete_emb_dim,
        reduce_parameter_action_dim=parameter_emb_dim)

    replay_buffer_embedding = utils.ReplayBuffer(state_dim, discrete_action_dim=1,
                                                 parameter_action_dim=1,
                                                 all_parameter_action_dim=parameter_action_dim,
                                                 discrete_emb_dim=discrete_emb_dim,
                                                 parameter_emb_dim=parameter_emb_dim,
                                                 max_size=int(1e1)
                                                 )

    replay_buffer = utils.ReplayBuffer(
        state_dim,
        discrete_action_dim=1,
        parameter_action_dim=1,
        all_parameter_action_dim=parameter_action_dim,
        discrete_emb_dim=discrete_emb_dim,
        parameter_emb_dim=parameter_emb_dim,
        max_size=int(2e4))

    agent_pre = PDQNAgent(
        state_dim, action_dims,
        batch_size=128,
        learning_rate_actor=0.001,
        learning_rate_actor_param=0.0001,
        epsilon_steps=1000,
        gamma=0.9,
        tau_actor=0.1,
        tau_actor_param=0.01,
        clip_grad=10.,
        indexed=False,
        weighted=False,
        average=False,
        random_weighted=False,
        initial_memory_threshold=500,
        use_ornstein_noise=False,
        replay_memory_size=10000,
        epsilon_final=0.01,
        inverting_gradients=True,
        zero_index_gradients=False,
        seed=args.seed)
    vae_load_model = args.load_model
    vae_save_model = args.save_model
    action_list = []
    if not vae_load_model:
        # ------Use random strategies to collect experience------
        max_steps = env.max_count
        total_reward = 0.
        returns = []
        sample_count = 0
        while sample_count < replay_buffer_embedding.max_size:
            state = env.reset()
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent_pre.act(state)
            all_action_parameters = torch.tensor(all_action_parameters)
            action_tmp = normalize_action(act, all_action_parameters, action_dims)
            episode_reward = 0.
            for j in range(max_steps):
                sample_count += 1
                ret = env.step(action_tmp)
                next_state, reward, terminal = ret
                next_state = np.array(next_state, dtype=np.float32, copy=False)
                next_act, next_act_param, next_all_action_parameters = agent_pre.act(
                    next_state)
                next_action_tmp = normalize_action(next_act, next_all_action_parameters, action_dims)
                state_next_state = next_state - state
                action_list.append(act)
                replay_buffer_embedding.add(
                    state,
                    act,
                    act_param,
                    all_action_parameters,
                    discrete_emb=None,
                    parameter_emb=None,
                    next_state=next_state,
                    state_next_state=state_next_state,
                    reward=reward,
                    done=terminal)
                act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
                action_tmp = next_action_tmp
                state = next_state
                episode_reward += reward
                if terminal:
                    break
            returns.append(episode_reward)
            total_reward += episode_reward
    print("discrete embedding:", action_rep.discrete_embedding())
    # ------AAE训练------
    VAE_batch_size = 1024
    calss_type = str(len(env.scenario.interesting_points) - 1)
    directory = (str(env.type) + "_" + str(calss_type))
    print(directory)
    if vae_load_model:
        print("load model")
        action_rep.load(directory)
        print("discrete embedding:", action_rep.discrete_embedding())
    else:
        file_name = str(env.type) + ".txt"
        if os.path.exists(file_name):
            os.remove(file_name)
        action_rep = AAE.Action_representation(
            state_dim=state_dim,
            action_dim=discrete_action_dim,
            parameter_action_dim=1,
            reduced_action_dim=discrete_emb_dim,
            reduce_parameter_action_dim=parameter_emb_dim)
        train_step = 1000
        aae_train(action_rep=action_rep, train_step=train_step,
                  replay_buffer=replay_buffer_embedding,
                  batch_size=VAE_batch_size, embed_lr=1e-4)
        if vae_save_model:
            action_rep.save(directory)
    # -------TD3训练------
    print("TD3 train")
    test_reward = 0
    state, done = env.reset(), False
    total_reward = 0.
    returns = []
    Reward = []
    Reward_100 = []
    max_steps = env.max_count
    cur_step = 0
    interval = 600000
    total_timesteps = 0
    t = 0
    while total_timesteps < args.max_timesteps:
        state = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        discrete_emb, parameter_emb = policy.select_action(state)
        # 探索
        if t < args.epsilon_steps:

            epsilon = args.expl_noise_initial - \
                      (args.expl_noise_initial - args.expl_noise) * (t / args.epsilon_steps)
        else:
            epsilon = args.expl_noise
        discrete_emb = (
                discrete_emb + np.random.normal(0, max_action * epsilon, size=discrete_emb_dim)
        ).clip(-max_action, max_action)
        parameter_emb = (
                parameter_emb + np.random.normal(0, max_action * epsilon, size=parameter_emb_dim)
        ).clip(-max_action, max_action)
        # select discrete action
        if not torch.is_tensor(discrete_emb):
            discrete_emb = torch.from_numpy(discrete_emb)
        discrete_emb = discrete_emb.float().reshape(1, -1).cpu()
        discrete_action = action_rep.select_discrete_action(
            discrete_emb)
        true_parameter_emb = parameter_emb
        parameter_action = action_rep.select_parameter_action(true_parameter_emb)
        action_tmp = utils.decode_discrete_action(discrete_action, action_dims)
        action = list()
        action.append(action_tmp[0])
        if action_tmp[0] == 1:
            parameter_action_tmp = 4 * (parameter_action + 1) / 2 + 4
        else:
            parameter_action_tmp = 5 * (parameter_action + 1) / 2 + 5
        parameter_action = parameter_action_tmp
        action.append(parameter_action_tmp[0])
        action.append(action_tmp[1])
        episode_reward = 0.
        if cur_step >= args.start_timesteps:
            parameter_relable_rate = policy.train(
                replay_buffer, args.batch_size)
        for i in range(max_steps):
            total_timesteps += 1
            cur_step = cur_step + 1
            ret = env.step(action)
            next_state, reward, terminal = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            state_next_state = next_state - state
            discrete_emb = discrete_emb.detach().numpy()
            replay_buffer.add(
                state,
                discrete_action=discrete_action,
                parameter_action=parameter_action,
                all_parameter_action=None,
                discrete_emb=discrete_emb,
                parameter_emb=parameter_emb,
                next_state=next_state,
                state_next_state=state_next_state,
                reward=reward,
                done=terminal)
            replay_buffer_embedding.add(
                state,
                discrete_action=discrete_action,
                parameter_action=parameter_action,
                all_parameter_action=None,
                discrete_emb=None,
                parameter_emb=None,
                next_state=next_state,
                state_next_state=state_next_state,
                reward=reward,
                done=done)
            next_discrete_emb, next_parameter_emb = policy.select_action(
                next_state)
            next_discrete_emb = (
                    next_discrete_emb + np.random.normal(0, max_action * epsilon, size=discrete_emb_dim)
            ).clip(-1, 1)
            next_parameter_emb = (
                    next_parameter_emb +
                    np.random.normal(
                        0,
                        max_action *
                        epsilon,
                        size=parameter_emb_dim))
            true_next_parameter_emb = next_parameter_emb
            next_parameter_action = action_rep.select_parameter_action(
                true_next_parameter_emb)
            if not torch.is_tensor(next_discrete_emb):
                next_discrete_emb = torch.from_numpy(next_discrete_emb)
            next_discrete_emb = next_discrete_emb.float().reshape(1, -1).cpu()
            next_discrete_action = action_rep.select_discrete_action(
                next_discrete_emb)
            next_action_tmp = utils.decode_discrete_action(next_discrete_action, action_dims)
            next_action = list()
            next_action.append(next_action_tmp[0])
            if action_tmp[0] == 1:
                next_parameter_action_tmp = 4 * (next_parameter_action + 1) / 2 + 4
            else:
                next_parameter_action_tmp = 5 * (next_parameter_action + 1) / 2 + 5
            next_action.append(next_parameter_action_tmp[0])
            next_action.append(next_action_tmp[1])
            next_parameter_action = next_parameter_action_tmp
            discrete_emb, parameter_emb, action, discrete_action, parameter_action = next_discrete_emb, \
                next_parameter_emb, next_action, next_discrete_action, next_parameter_action
            state = next_state
            if cur_step >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)
            episode_reward += reward
            if total_timesteps % args.eval_freq == 0:
                tmp_test_reward, finished = evaluate(
                    env, policy, action_rep, log, episodes=50)
                if finished == 1 and tmp_test_reward > test_reward:
                    test_reward = tmp_test_reward
                    torch.save(policy.actor, '%s/actor.pt' % directory)
            if terminal:
                break
        t = t + 1
        returns.append(episode_reward)
        total_reward += episode_reward

        # vae 训练
        if t % interval == 0:
            print("AAE", t)
            aae_train(action_rep=action_rep, train_step=6000, replay_buffer=replay_buffer_embedding,
                                        batch_size=VAE_batch_size, embed_lr=1e-4)


def aae_train(
        action_rep,
        train_step,
        replay_buffer,
        batch_size,
        embed_lr):
    initial_losses = []
    for counter in range(int(train_step) + 10):
        losses = []
        state, discrete_action, parameter_action = replay_buffer.sample(batch_size, 0)
        aae_loss = action_rep.unsupervised_loss(state, discrete_action.reshape(
            1, -1).squeeze().long(), parameter_action, batch_size, embed_lr)
        aae_loss = aae_loss.cpu().detach().numpy()
        losses.append(aae_loss)
        initial_losses.append(np.mean(losses))
        if len(initial_losses) >= train_step and np.mean(
                initial_losses[-5:]) + 1e-5 >= np.mean(initial_losses[-10:]):
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--policy", default="P-TD3")
    parser.add_argument("--env", default='Platform-v0')  # platform goal HFO
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int)
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=2000, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument(
        "--max_episodes",
        default=100000,
        type=int)  # Max time steps to run environment
    parser.add_argument(
        "--max_embedding_episodes",
        default=1e5,
        type=int)  # Max time steps to run environment
    # Max time steps to run environment for
    parser.add_argument("--max_timesteps", default=30000000, type=float)

    # Max time steps to epsilon environment
    parser.add_argument("--epsilon_steps", default=5000, type=int)
    # Std of Gaussian exploration noise 1.0
    parser.add_argument("--expl_noise_initial", default=1)
    # Std of Gaussian exploration noise 0.1
    parser.add_argument("--expl_noise", default=0.4
                        )
    parser.add_argument(
        "--relable_steps",
        default=1000,
        type=int)  # Max time steps relable
    parser.add_argument("--relable_initial", default=1.0)  #
    parser.add_argument("--relable_final", default=0.0)  #

    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.995)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.10)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.10)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=30, type=int)
    # Save model and optimizer parameters
    parser.add_argument("--save_model", default=True
                        )
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model", default=False)
    args = parser.parse_args()
    for i in range(0, 1):
        args.seed = i
        for dis_emb in range(14, 40, 2):
            for para_emb in range(6, 50, 2):
                run(args, dis_emb, para_emb)

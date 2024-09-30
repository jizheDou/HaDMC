# ！/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/3/13 19:21
# @Author  : doujizhe
# @File    : world
# @Software: PyCharm
import copy
import random
import time
from math import sqrt, acos
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import simpy
import torch
from map.geometry import Point
from map.geometry import InterestingPoint
from map.geometry import ChargingTable
from vehicle.vehicle import Uav
from vehicle.vehicle import Robot
from map.geometry import Map
from scenario.scenario import CircleScenario


class World:
    """simulate the world.it will change because action"""

    def __init__(self, scenario):
        self.scenario = scenario

    def reset(self):
        pass

    def step(self, action):
        pass


class CircleWorld(World):

    def __init__(self, scenario):
        super().__init__(scenario)
        self.state = None
        self.env = self.scenario.env
        self.action_space = None
        self.observation_space = None
        self.penalty_factors = None

    def reset(self):
        return self.scenario.get_state()

    def action_limitation(self, action):
        pass

    def step(self, action):
        """
        agent takes action make the world change
        :param action: take action, two agents: charger, uav.
        For uav, actions insist of action ' s direction or action ' s kind and next position
        :return:
        """
        reward = self.calculate_reward(action)
        next_state = self.scenario.step(action)
        done = self.is_finished()
        return reward, next_state, done

    def calculate_reward(self, action):
        """
        calculate action ' s reward
        :param action: agent takes action
        :return: reward
        """
        pass

    def is_finished(self):
        """
        tasks finished or not finished
        :return: True or False
        """
        return False


class TestWorld(CircleWorld):

    def __init__(self, scenario):
        super(TestWorld, self).__init__(scenario)
        self.current_energy = 0
        self.finished = 0
        self.count = 0
        self.goal = None
        self.distance_threshold = 10
        self.action_bound = 10
        # self.env.process(self.run())
        self.reset()
        self.type = 0
        self.max_count = 100
        self.success_count = 0
        self.start_time = 0
        self.end_time = 0
        self.sum_time = 0
        self.charging_count = 0

    def generate_interesting_points(self, point_num: int):
        """
        generate interesting points according to reset points.
        :param point_num: generated points' number
        :return: generated points' list
        """
        m = 400
        n = 400
        a = 400
        b = 400
        theta = np.linspace(3 * np.pi / 2, - np.pi / 2, point_num + 2, endpoint=True)
        x = m + a * np.cos(theta) + np.random.randint(0, 10, point_num + 2) + 250
        y = n + b * np.sin(theta) + np.random.randint(0, 10, point_num + 2) + 100
        num = 0
        interesting_points = list()
        for i in range(1, point_num + 1):
            if i % 2 != 0:
                point = Point(x[i] - 50, y[i] + 10)
            else:
                if x[i] < 500:
                    point = Point(x[i] + random.randint(-250, -230), y[i] + random.randint(0, 10))
                else:
                    point = Point(x[i] + random.randint(-250, -230), y[i] + random.randint(-10, 0))
            interesting_points.append(InterestingPoint(1, num, point))
            num += 1
        return interesting_points

    def generate_charging_tables_around_interesting_points(
            self, interesting_points: list, length):
        """
        generate random charging tables around interesting points.
        :param length: the random length in axis relative to interesting points
        :param interesting_points: the list of interesting point
        :return: charging tables
        """
        charging_tables = list()
        for i in range(2, (len(interesting_points)), 1):
            if i % 3 != 0 and i % 11 != 0:
                continue
            charging_table_position = Point(
                interesting_points[i].x +
                random.randint(
                    length / 5,
                    length / 2),
                interesting_points[i].y +
                random.randint(
                    length / 5,
                    length / 2),
                0)
            charging_table = ChargingTable(
                2, i, "Charging Station", charging_table_position, 20, 20)
            charging_tables.append(charging_table)
        return charging_tables

    def generate_random_grid_points(self, row: int, col: int, map: Map):
        charging_tables = list()
        num = 0
        for i in range(0, map.size[0], int(map.size[0] / row) + 1):
            for j in range(0, map.size[1], int(map.size[1] / col) + 1):
                if i == 0 and j == 1 * (int(map.size[1] / col) + 1):
                    continue
                y = random.randint(i, i + int(map.size[0] / (row * 4))) + 50
                x = random.randint(j, int(j + map.size[1] / (col * 4))) + 150
                charging_table_position = Point(x, y)
                charging_table = ChargingTable(
                    2, num, "Charging Station", charging_table_position, 20, 20)
                charging_tables.append(charging_table)
                num += 1
        return charging_tables

    def reset(self):
        env = simpy.Environment()
        # tk1 = tk.Tk()
        sim_log = "uav1"
        uav_id = 1
        uav_init_position = Point(
            500, 10, 0)
        uav_env = env
        self.type = 2
        uav_velocity = 25  # 20m/s, initial velocity is 72 km/h, 20m/s
        uav_energy = 60
        uav_log = sim_log
        uav_charging_velocity = 6
        uav_time_scale = 1
        uav = Uav(uav_id, 'uav', uav_init_position, uav_env, uav_velocity,
                  uav_energy, uav_log, uav_charging_velocity, uav_time_scale)
        charger_velocity = 10
        charger_init_position = uav_init_position
        charger = Robot(
            1,
            "charger",
            charger_init_position,
            env,
            charger_velocity,
            sim_log,
            1)
        map = Map([1000, 1000])
        interesting_points = self.generate_interesting_points(40)
        if self.type == 2:
            charging_tables = self.generate_random_grid_points(4, 4, map)
        elif self.type == 1:
            charging_tables = self.generate_charging_tables_around_interesting_points(interesting_points, 100)
        charging_depot = ChargingTable(2, len(charging_tables), "Charging Station", uav_init_position, 20, 20)
        charging_tables.append(charging_depot)
        interesting_points.append(InterestingPoint(1, len(interesting_points), Point(uav.position.x, uav.position.y)))

        self.scenario = CircleScenario(
            's1',
            env=env,
            map=map,
            uav=uav,
            charger=charger,
            interesting_points=interesting_points,
            charging_tables=charging_tables)
        self.state = self.scenario.get_state()
        self.count = 0
        self.charging_count = 0
        self.start_time = float(0)
        self.end_time = float(0)
        self.sum_time = float(0)
        self.sum_reward = 0
        self.finished = 0
        return self.state

    def step(self, action):
        """
         We assume interesting points are front of charging tables.
        :param action: action[0] is the uav and charger decision, action[1] is hovering time.
        :return: next state, reward, done
        """
        utility_time = 0
        possible_action_list, min_dist_index = self.possible_actions(action)
        reward = -self.calculate_reward(action)
        if action[0] == 1:
            if self.scenario.visited_interesting_count < len(
                    self.scenario.interesting_points):
                visited_point = self.scenario.interesting_points[self.scenario.visited_interesting_count]
                utility_time = min(action[1].item(), visited_point.max_visited_time)
        self.current_energy = self.scenario.uav.battery.current_energy
        self.state = self.scenario.step(action)
        if action[0] == 1:
            self.end_time = self.scenario.uav.count_time
            if self.scenario.visited_interesting_count == len(self.scenario.interesting_points):
                utility_time = 0
            if self.scenario.uav.battery.current_energy <= 0:
                utility_time = 0
            interval = self.end_time - self.start_time - utility_time
            interval = max(1, interval)
            self.sum_time += interval
            self.start_time = self.end_time
            if reward > 0:
                reward = 1 * reward * max(1, self.scenario.visited_interesting_count /
                                          2
                                          )
                if action[2] in possible_action_list and self.scenario.visited_interesting_count < len(
                        self.scenario.interesting_points):
                    reward += ((2 * max(1, self.scenario.visited_interesting_count / 1)
                                ) /

                               (max(1, (
                                       (self.scenario.charging_tables[action[2]].distance_to(
                                           self.scenario.interesting_points[
                                               self.scenario.visited_interesting_count - 1]) *
                                        self.scenario.charging_tables[action[2]].distance_to(
                                            self.scenario.interesting_points[
                                                self.scenario.visited_interesting_count])) / self.scenario.uav.velocity ** 2
                               )
                                    )
                                )
                               )
                self.sum_reward += reward
                reward = 1 * reward / interval
                reward = reward * 5 * self.task_finished_ratio() / self.sum_time
        self.count += 1
        if action[0] == 0:
            if self.sum_time > 0 and self.task_finished_ratio() > 0:
                reward = reward * 5 * self.task_finished_ratio() / self.sum_time
            self.charging_count += 1
            if action[2] not in possible_action_list:
                reward = 0
        done = self.is_finished(action, possible_action_list)
        if done == 1 and self.count < self.max_count - 1:
            reward += 60 * max(1,
                               self.scenario.visited_interesting_count / 3) * self.task_finished_ratio() / self.sum_time
            self.finished = 1
            self.success_count += 1
        if (done == -1 or self.count >= self.max_count + 1 or
                self.charging_count >= self.max_count - len(self.scenario.interesting_points)):
            reward += -20 * self.penalty()
            done = True
        return self.state, reward, done

    def penalty(self):
        """
        the distance between the max of information and the gotten information
        :return: penalty, the non-completion rate of all tasks.
        """
        penalty = 0
        for interesting_point in self.scenario.interesting_points:
            if interesting_point.visited_time < interesting_point.max_visited_time:
                penalty += (interesting_point.max_visited_time -
                            interesting_point.visited_time) / interesting_point.max_visited_time
        penalty = penalty / len(self.scenario.interesting_points)
        return penalty

    def calculate_reward(self, action):
        """
        consider the reward without flying time
        :param action:
        :return:
        """
        reward = 0
        # visited all interesting points but not finished observation task
        if self.scenario.visited_interesting_count >= len(
                self.scenario.interesting_points):
            if action[2] == len(self.scenario.charging_tables) - 1:
                reward += 0
            else:
                reward += 0
            return reward
        if self.count == 0 and action[0] == 1:
            reward += 0
        visited_point = self.scenario.interesting_points[self.scenario.visited_interesting_count]
        to_next_visited_point_dist = visited_point.distance_to(self.scenario.uav.position)
        flying_time = to_next_visited_point_dist / self.scenario.uav.velocity
        consumed_energy = flying_time * self.scenario.uav.time_scale
        if action[0] == 0:
            chosen_charging_point = self.scenario.charging_tables[action[2]]
            charging_flying_time = chosen_charging_point.distance_to(
                self.scenario.uav.position) / self.scenario.uav.velocity
            charging_flying_consumed_energy = charging_flying_time * self.scenario.uav.time_scale
            charger_to_charging_point_dist = self.scenario.charger.position.distance_to(chosen_charging_point)
            charger_running_time = charger_to_charging_point_dist / self.scenario.charger.velocity
            charging_point_to_visited_point_dist = visited_point.distance_to(chosen_charging_point)
            charging_point_to_visited_point_time = (charging_point_to_visited_point_dist /
                                                    self.scenario.uav.velocity)
            charging_time = ((self.scenario.uav.battery.max_energy - (self.scenario.uav.battery.current_energy -
                                                                      charging_flying_consumed_energy)) /
                             self.scenario.uav.charging_velocity)
            reward += ((-1 * max(1, self.scenario.visited_interesting_count / 6) * min(action[1], charging_time)) /
                       (max(1, self.scenario.uav.battery.current_energy / 10) *
                        (max(1, action[1] - charging_time) *
                         (max(1, max(charging_flying_time, charger_running_time)) + charging_point_to_visited_point_time
                          )
                         )
                        )
                       )
            return reward
        if action[0] == 1:
            # repeatedly visit finished interesting point
            if visited_point.visited_time >= visited_point.max_visited_time and \
                    (self.scenario.visited_interesting_count != 0):
                reward += max(1, action[1])
                return reward
            else:
                # visited time < observation's minimum
                if action[1] + \
                        visited_point.visited_time < visited_point.min_visited_time:
                    reward += 1 * (visited_point.min_visited_time - action[1])
                    return reward
                # observation's minimum < visited time < observation's maximum
                if visited_point.min_visited_time <= \
                        action[1] + visited_point.visited_time <= visited_point.max_visited_time:
                    reward += -1 * (min(action[1], action[1] + visited_point.visited_time))
                    return reward
                # observation's maximum < visited time
                if action[1] + \
                        visited_point.visited_time > visited_point.max_visited_time:
                    reward += -1 * (visited_point.max_visited_time - visited_point.visited_time) / \
                              max(1, action[1] - visited_point.max_visited_time)
                    return reward
        return reward

    def possible_actions(self, action):
        charging_point_list = []
        visited_time = 0
        if action[0] == 1:
            visited_time = self.scenario.interesting_points[self.scenario.visited_interesting_count].max_visited_time
        min_dist = self.scenario.charging_tables[0].distance_to(self.scenario.uav.position)
        index = 0
        for i in range(len(self.scenario.charging_tables)):
            if action[0] == 0:
                to_charging_point_dist = self.scenario.charging_tables[i].distance_to(self.scenario.uav.position)
                consumed_energy = to_charging_point_dist / self.scenario.uav.velocity
            else:
                to_interesting_point_dist = self.scenario.interesting_points[
                    self.scenario.visited_interesting_count].distance_to(self.scenario.uav.position)
                to_charging_point_dist = self.scenario.charging_tables[i].distance_to(
                    self.scenario.interesting_points[self.scenario.visited_interesting_count])
                consumed_energy = (to_charging_point_dist + to_interesting_point_dist) / self.scenario.uav.velocity
            if action[0] == 1:
                if consumed_energy + 8 * self.scenario.uav.time_scale <= self.scenario.uav.battery.current_energy and consumed_energy <= 0.5 * self.scenario.uav.battery.max_energy:
                    charging_point_list.append(i)
                if to_charging_point_dist < min_dist:
                    index = i
            else:
                if consumed_energy <= self.scenario.uav.battery.current_energy and consumed_energy <= 0.2 * self.scenario.uav.battery.max_energy:
                    charging_point_list.append(i)
                if to_charging_point_dist < min_dist:
                    index = i
        return charging_point_list, index

    def is_finished(self, action, possible_action_list: list) -> object:
        done = True
        if self.scenario.uav.battery.current_energy < 0:
            return -1
        visited_count = 0
        for point in self.scenario.interesting_points:
            if visited_count < self.scenario.visited_interesting_count:
                if point.is_visited is False:
                    return -1
            visited_count += 1
            if not point.is_visited:
                done = False
                break
        if self.scenario.interesting_points[-1].is_visited == 1:
            if done is False:
                return -1
            else:
                return True
        if done is True:
            dist = self.scenario.charger.position.distance_to(self.scenario.depot)
            if dist <= 10:
                return True
            else:
                return False
        return done

    def task_finished_ratio(self):
        finished_time = 0
        for point in self.scenario.interesting_points:
            finished_time += min(point.visited_time, point.max_visited_time)
        return finished_time

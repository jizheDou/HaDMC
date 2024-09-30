# ！/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/7/2 20:26
# @Author  : doujizhe
# @File    : GreedyObservation
# @Software: PyCharm
from world.world import World, TestWorld


class GreedyObservation:

    def __init__(self, world: World):
        """

        :param world:
        """
        self.objective = 0
        self.world = world
        self.interesting_points = self.world.scenario.interesting_points
        self.charging_tables = self.world.scenario.charging_tables
        self.uav = self.world.scenario.uav
        self.charger = self.world.scenario.charger
        self.action_list = list()
        self.sum_observation_time = 0
        self.sum_charging_time = 0
        self.sum_waiting_time = 0
        self.sum_time = 0
        self.sum_flying_time = 0

    def nearest_charging_table(self, point):
        min_dist = point.distance_to(self.charging_tables[0])
        nearest_charging_table = self.charging_tables[0]
        count = -1
        pos = 0
        for charging_table in self.charging_tables:
            count += 1
            dist = point.distance_to(charging_table)
            if min_dist > dist:
                nearest_charging_table = charging_table
                min_dist = dist
                pos = count
        return nearest_charging_table, pos

    def run(self):
        last_leave_energy = 0
        current_point = self.uav.position
        sum_observing_time = 0
        sum_flying_time = 0
        sum_task_finished_ratio = 0
        sum_charging_time = 0
        sum_waiting_time = 0
        sum_observation_time = 0
        charging_time = 0
        current_energy = self.uav.max_energy
        current_charger_point = self.uav.position
        current_charging_table = self.interesting_points[-1]
        current_point_to_charging_dist = 0
        done = False
        count = 0
        flag = 0
        action = [1, 0, 0]
        last_pos = -1
        next_pos = -1
        update_pos = -1
        charging_count = 0
        while not done:
            next_point = self.interesting_points[count]
            next_charging_table, next_pos = self.nearest_charging_table(next_point)
            point_to_point_dist = current_point.distance_to(next_point)
            point_to_charging_dist = next_point.distance_to(next_charging_table)
            dist = point_to_point_dist + point_to_charging_dist
            flying_time = dist / self.uav.velocity
            leave_energy = (current_energy - flying_time
                            - next_point.max_visited_time * self.uav.time_scale)
            if leave_energy < 0:
                if flag == 1:
                    print("No solution!")
                    exit()
                flag += 1
                charging_count += 1
                dist = current_point_to_charging_dist
                flying_time = dist / self.uav.velocity
                current_energy -= flying_time * self.uav.time_scale
                sum_waiting_time += max(flying_time, current_charger_point.distance_to(current_charging_table)
                                        / self.charger.velocity)
                current_charger_point = current_charging_table
                sum_flying_time += flying_time
                charging_time = (self.uav.max_energy - current_energy) / self.uav.charging_velocity
                sum_charging_time += charging_time
                current_energy = self.uav.max_energy
                current_point = current_charging_table
                action = [0, charging_time, last_pos]
                update_pos = last_pos
            else:
                flag = 0
                dist = point_to_point_dist
                flying_time = dist / self.uav.velocity
                sum_flying_time += flying_time
                observation_time = next_point.max_visited_time
                sum_observation_time += observation_time
                current_energy = current_energy - flying_time * self.uav.time_scale - next_point.max_visited_time * self.uav.time_scale
                current_point = next_point
                current_charging_table = next_charging_table
                last_pos = next_pos
                current_point_to_charging_dist = point_to_charging_dist
                count += 1
                action = [1, observation_time, update_pos]
            self.action_list.append(action)
            if count == len(self.interesting_points):
                self.action_list[-1][2] = -1
                break
        sum_waiting_time += max(0,
                                current_charger_point.distance_to(current_point) / self.charger.velocity - flying_time)
        sum_time = (sum_flying_time + sum_charging_time + sum_waiting_time)
        objective = sum_observation_time / sum_time
        self.sum_observation_time = sum_observation_time
        self.sum_waiting_time = sum_waiting_time
        self.sum_charging_time = sum_charging_time
        self.sum_flying_time = sum_flying_time
        self.sum_time = sum_time
        self.objective = objective
        return objective, self.action_list

    def print_time(self):
        return (self.sum_observation_time, self.sum_charging_time, self.sum_waiting_time, self.sum_flying_time,
                self.sum_time, self.objective)

import sys

import numpy as np
import torch

sys.path.append('../')
from map.geometry import Point
from map.geometry import Depot
from vehicle.component import EnergyComponent
from map.geometry import Map
import random


def circle(r, x0, y0, x):
    """
    define a quarter circle
    :param x0: center coordinate x
    :param y0: center coordinate y
    :param r: radium of circle
    :param x: generate point's x
    :return: point's y
    """
    y = y0 + np.sqrt(r ** 2 - (x - x0) ** 2)
    return y


def map(action):
    """according to the number of action give the function to execute
    actions include two dimensions, guide deciding whether charging or
    hovering or running and destination number or direction
    """


class Scenario:
    """This class is to build the basic environment"""

    def __init__(
            self,
            name,
            env,
            map=None,
            points: Point = None,
            facilities=None):
        if map is None:
            self.map = Map([1000, 1000])
        else:
            self.map = map
        if facilities is None:
            facilities = []
        if points is None:
            points = []
        if points is None:
            points = []
        self.name = name
        self.map = map
        self.points = points
        self.env = env
        self.facilities = facilities
        self.state = None
        self.observation_space_dimension = 0

    def generate_points(self):
        pass

    def generate_facilities(self):
        pass

    def plot_scenario(self):
        pass

    def state(self):
        pass


class CircleScenario(Scenario):
    """The circular scenario where UAV visits all PoIs in a circular trajectory.
    simulate the real world which can be changed by taking action"""

    def __init__(
            self,
            name,
            env,
            map=None,
            interesting_points=None,
            facilities=None,
            charging_tables=None,
            uav=None,
            charger=None):
        super(
            CircleScenario,
            self).__init__(
            name,
            env,
            map,
            interesting_points,
            facilities)

        if charging_tables is None:
            self.charging_tables = list()
        else:
            self.charging_tables = charging_tables
        self.uav = uav
        if interesting_points is None:
            self.interesting_points = list()
        else:
            self.interesting_points = interesting_points
        self.charger = charger
        if uav is None:
            self.depot = [0, 0, 0]
            self.time_scale = 1
            self.charging_velocity = 1
        else:
            self.time_scale = self.uav.time_scale
            self.charging_velocity = self.uav.charging_velocity
            self.depot = Depot(1, 0, self.uav.position, 'depot')
        self.visited_interesting_count = 0
        self.charging_station_count = 0
        self.charging_count = 0
        self.count = 0
        self.max_finished_percent = 0.8 + random.random() * 0.01
        self.sum_charging_time = 0
        self.sum_observation_time = 0
        self.sum_flying_time = 0
        self.sum_waiting_time = 0
        # self.state = self.get_state()

    @property
    def max_observation_time(self):
        time = 0
        for i in range(0, len(self.interesting_points) - 1):
            time += self.interesting_points[i].max_visited_time
        return time


    @property
    def enabled_visited_points(self):
        points = list()
        for point in self.interesting_points:
            points.append(point)
        return points

    @property
    def action_space_size(self):
        """get action space size which is equals number of depot and charging tables and interesting points.
        """
        action_space_size = 2
        action_space_size += len(self.charging_tables)
        return action_space_size, len(self.charging_tables)

    def generate_facilities(self):
        """generate kinds of facilities, such as planes"""
        pass

    def reset(self):
        """reset scenario to initial state"""
        number = 10
        ratio = 0.4
        width = 0.5
        length = 0.5

    def update_battery(self, action, battery: EnergyComponent):
        if battery.current_energy < 0:
            return battery
        if torch.is_tensor(self.uav.count_time):
            self.uav.count_time = self.uav.count_time.item()
        if action[0] == 0 and \
                self.uav.position.distance_to(self.charger.position) <= 10:
            battery.add(action[1] * self.charging_velocity)
        else:
            battery.consume(action[1] * self.time_scale)
        return battery

    def update_charger_position(self, position: Point, action, flying_time):
        new_charger_position = Point(self.charging_tables[action[2]].x,
                                     self.charging_tables[action[2]].y, 0)
        running_dist = position.distance_to(new_charger_position)
        need_running_time = running_dist / self.charger.velocity
        if flying_time == -1:
            new_charger_position = Point(self.charging_tables[-1].x,
                                         self.charging_tables[-1].y, 0)
            running_dist = position.distance_to(new_charger_position)
            return new_charger_position, need_running_time
        if action[0] == 0:
            return new_charger_position, need_running_time
        running_end_time = self.uav.count_time + flying_time + action[1].item()
        get_running_time = flying_time + action[1].item()
        if self.charger.count_time + need_running_time >= running_end_time:
            if need_running_time == 0:
                new_charger_position.x = position.x
                new_charger_position.y = position.y
            else:
                new_charger_position.x = position.x + (new_charger_position.x - position.x) * min(max(0,
                                     get_running_time / need_running_time), 1)
                new_charger_position.y = position.y + (new_charger_position.y - position.y) * min(max(0,
                                     get_running_time / need_running_time), 1)
        return new_charger_position, get_running_time

    def update_uav_position(self, position: Point, action):
        """
        update uav position, if value of action[0] is 0, represents uav keep static.
        if value of action[0] is 1, represents uav goes to next interesting point.
        :param position:
        :param action:
        :return:
        """
        if action[0] == 1:
            new_position = self.enabled_visited_points[self.visited_interesting_count]
            dist = position.distance_to(new_position)
            flying_time = dist / self.uav.velocity
            self.visited_interesting_count += 1
            return new_position, flying_time
        else:
            new_position = self.charging_tables[action[2]]
            dist = position.distance_to(new_position)
            flying_time = dist / self.uav.velocity
            return new_position, flying_time

    def step(self, action):



        """change state of scenario step by step, action[0] is the next point, action[1] is hovering time."""
        if action[0] == 1:
            action[1] = min(torch.from_numpy(np.array(self.interesting_points[self.visited_interesting_count].max_visited_time)), action[1])
            self.sum_observation_time += int(action[1])
        self.count += 1
        # finished condition
        # if action[0] == 1:
        #     if self.visited_interesting_count >= len(self.interesting_points):
        #         self.visited_interesting_count += 1
        #         self.charger.position, charger_running_time = self.update_charger_position(
        #             self.charger.position, action, -1)
        #         self.charger.count_time += charger_running_time
        #         return self.get_state()
        # flying's time, running's time and energy update

        self.uav.position, flying_time = self.update_uav_position(
            self.uav.position, action)
        self.uav.battery.consume(flying_time * self.time_scale)
        if action[0] == 0:
            max_charging_time = (self.uav.battery.max_energy - self.uav.battery.current_energy) / self.uav.charging_velocity
            action[1] = min(torch.from_numpy(np.array(max_charging_time)), action[1])
            self.sum_charging_time += action[1]
        self.uav.count_time += flying_time
        self.sum_flying_time += flying_time
        # visit last PoI (depot)
        if self.visited_interesting_count == len(self.interesting_points):
            dist = self.charger.position.distance_to(self.uav.position)
            self.charger.position = self.uav.position
            back_time = dist / self.charger.velocity
            self.uav.count_time += max(0, back_time - flying_time)
            self.sum_flying_time += max(0, back_time - flying_time)
            if action[0] == 1 and self.uav.battery.current_energy >= 0:
                self.interesting_points[self.visited_interesting_count -
                                        1].is_visited = 1
            return self.get_state()
        self.charger.position, charger_running_time = self.update_charger_position(
            self.charger.position, action, flying_time)
        self.charger.count_time += charger_running_time
        # observing or charging 's energy update
        self.uav.battery = self.update_battery(action, self.uav.battery)

        # print("uav energy", self.uav.battery.current_energy)
        if self.uav.battery.current_energy < 0:
            return self.get_state()
        if action[0] == 1:
            # observing 's time update
            self.uav.count_time += action[1].item()
            self.interesting_points[self.visited_interesting_count -
                                    1].visited_time += action[1].item()
            if self.interesting_points[self.visited_interesting_count - 1].visited_time >= \
                    self.interesting_points[self.visited_interesting_count - 1].min_visited_time:
                self.interesting_points[self.visited_interesting_count -
                                        1].is_visited = 1
        if action[0] == 0:
            self.charging_tables[action[2]].visited_time += action[1].item()
            self.charger.count_time += action[1].item()
            self.uav.count_time += action[1].item()
            current_time = max(self.charger.count_time, self.uav.count_time)
            self.sum_waiting_time += max(0, self.charger.count_time - self.uav.count_time)
            self.charger.count_time = current_time
            self.uav.count_time = current_time
        if self.charger.count_time < self.uav.count_time:
            self.charger.count_time = self.uav.count_time
        return self.get_state()

    def get_state(self):
        """name: Depot: 0 , uav: 1, charger: 2, charging station: 3, interesting point: 4
        PoI: name, position, is visited
        Charging table: name, position
        uav: name, position, energy, (velocity)
        charger: name, position, (velocity)
        :return:
        """
        state = list()

        for point in self.interesting_points:
            interesting_point_state = [4,
                                       point.num / 50,
                                       point.x * 10 / self.map.size[0],
                                       point.y * 10 / self.map.size[0],
                                       point.z * 10 / self.map.size[0],
                                       point.min_visited_time,
                                       point.max_visited_time,
                                       point.is_visited,
                                       point.visited_time,
                                       min(1,
                                           point.visited_time / point.max_visited_time),
                                       max(0,
                                           point.max_visited_time - point.visited_time)]
            if torch.is_tensor(interesting_point_state):
                interesting_point_state = interesting_point_state.detach().numpy()
            if torch.is_tensor(interesting_point_state[10]):
                interesting_point_state[10] = interesting_point_state[10].detach().numpy()
            interesting_point_state[10] = np.round(interesting_point_state[10], 3)
            interesting_point_state = np.array(interesting_point_state)
            state.append(interesting_point_state[[1, 2, 3, 5, 6, 7, 8, 9, 10]])
            # self.observation_space_dimension += len(interesting_point_state)
        for point in self.charging_tables:
            charging_table_state = [
                3,
                point.num,
                point.x * 10 / self.map.size[0],
                point.y * 10 / self.map.size[0],
                point.z * 10 / self.map.size[0],
                point.visited_time]
            charging_table_state = np.array(charging_table_state)
            state.append(charging_table_state[[2, 3, 5]])
            # self.observation_space_dimension += len(charging_table_state)
        uav_state = self.uav.get_state()
        uav_state[0] = 1
        uav_state = np.array(uav_state)
        uav_state[2] = uav_state[2] / self.map.size[0]
        uav_state[3] = uav_state[3] / self.map.size[0]
        uav_state[5] = uav_state[5] / 10
        uav_state[6] = round(uav_state[6], 2) / 10
        uav_state[7] = uav_state[7] / 10
        uav_state[8] = uav_state[8] / 10
        state.append(uav_state[[1, 2, 3, 5, 6, 7, 8, 9]])
        charger_state = self.charger.get_state
        charger_state[0] = 2
        charger_state[2] = charger_state[2] * 10 / self.map.size[0]
        charger_state[3] = charger_state[3] * 10 / self.map.size[0]
        charger_state = np.array(charger_state)
        state.append(charger_state[[2, 3]])
        depot_state = [0, self.depot.x, self.depot.y, self.depot.is_visited]
        # depot_state = np.array(depot_state)
        # state.append(depot_state)
        count = [self.count, self.visited_interesting_count]
        count = np.array(count)

        state.append(count)
        init_state = np.array([])
        self.observation_space_dimension = sum([i.__len__() for i in state])
        for next_state in state:
            init_state = np.hstack((init_state, next_state))
        state = init_state
        return state


def name():
    """Depot: 0 , uav: 1, charger: 2, charging station: 3, interesting point: 4"""
    pass

# 运动模式  速度 时间 运动到的地点
# event 随机游走

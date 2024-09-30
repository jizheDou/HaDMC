#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/26 21:16
# @Author  : S.G.D.
# @File    : vehicle
from abc import ABCMeta, abstractmethod, ABC
from typing import Dict
import simpy
from vehicle.logger import SimLogger
from map.geometry import distance
from map.geometry import Point
from vehicle.component import Component, EnergyComponent
from event.event import MovingSphereEvent


def interrupt(reason: str):
    simpy.Interrupt(reason)


class Vehicle(metaclass=ABCMeta):
    """ This class is the base class for other specific vehicles. """

    def __init__(self, vid: int, name: str, position: Point, env: simpy.Environment(), velocity: float, log_filename,
                 time_scale: float):
        self.id = vid
        self.name = name
        self.position = position
        self.components = dict()
        self.env = env
        self.velocity = velocity
        self.time_scale = time_scale
        self.log = SimLogger("%s" % log_filename)
        self.state = "static"
        self.moving_event_count = 0
        self.fixed_event_count = 0
        self.start_time = 0
        self.end_time = 0
        self.run_time = 0
        #self.state = self.get_state

    def __hash__(self):
        return hash((self.id, self.name))

    def __eq__(self, other):
        return self.id == other.id and self.name == other.name

    def set_components(self, components: Dict[str, Component]):
        """
        Put components into the vehicle.
        param components: a dictionary of {component's name : the corresponding object}
        :return:
        """
        self.components = components

    @abstractmethod  # send message
    def send(self):
        pass

    def calculate_run_time(self, pos1: Point, pos2: Point) -> float:
        """
        Calculate the running time from pos1 to pos2
        :param pos1:
        :param pos2:
        :return:
        """
        dis = distance(pos1, pos2)
        return dis / self.velocity

    def convert_to(self, state, duration_time) -> float:
        """
        convert to the new state
        :param state: the new state
        :param duration_time: the duration_time of new state
        :param self.time_scale the scale of time
        :return: float
        """
        self.state = state
        start_time = self.env.now
        while duration_time > 0:
            try:
                yield self.env.timeout(self.time_scale)
                duration_time = duration_time - self.time_scale
                self.update_energy(self.state)
                self.position = self.update_position(self.state, self.position)
                self.log_data(self.state)
            except simpy.Interrupt as i:
                end_time = self.env.now
                duration_time = duration_time - (start_time - end_time)
                print(i)
                return duration_time
        return 0

    @abstractmethod
    def update_energy(self, state):
        pass

    @abstractmethod
    def log_data(self, state) -> None:
        """
        record data in different state
        :param state: current state
        :return: None
        """
        pass

    def update_position(self, state, init_dist, position: Point):
        """
        update the position
        :param state: current state
        :param position: the current position
        :param init_dist:distance between current position and destination
        :return:
        """
        pass

    def has_arrived(self, init_dist):
        pass

    def run_to(self, destination: Point, init_dist, state="running"):
        """
        run to the point from the self. position
        :param init_dist:distance between current position and destination
        :param state: change running to state
        :param destination: the destination
        :return:
        """
        moving_event = MovingSphereEvent(self.position, self.moving_event_count, self.name,
                                         self.env, state, self.time_scale)
        self.moving_event_count += 1
        try:
            while True:
                yield self.env.timeout(1)
                init_dist = self.run_step(moving_event, destination, init_dist, state)
                print(init_dist)
                if moving_event.is_including(destination) or init_dist <= 0:
                    print("Finished")
                    print(self.env.now)
                    break
        except simpy.Interrupt as interrupt_reason:
            print(interrupt_reason)

    def run_step(self, moving_event, destination: Point, init_dist, state="running"):
        """
        run to the point from the self. position
        :param moving_event: bind a moving sphere event
        :param init_dist:distance between current position and destination
        :param state: change running to state
        :param destination: the destination
        :return: leave distance
        """
        try:
            # self.env.process(moving_event.move())
            print(moving_event.position.x, moving_event.position.y)
            init_dist = self.update_position(state, init_dist, destination)
            return init_dist
        except simpy.Interrupt as interrupt_reason:
            print(interrupt_reason)
        return init_dist

    @abstractmethod  # receive message
    def receive(self):
        pass

    def get_state(self):
        """get the state of object"""
        pass


class Uav(Vehicle):
    """This class is UAV"""

    def __init__(
            self,
            vid,
            name,
            position,
            env: simpy.Environment(),
            velocity: float,
            max_energy: float,
            sim_log,
            charging_velocity,
            time_scale):
        """

        :param vid: the id of UAV
        :param name: the name of UAV
        :param position: the position of UAV
        :param env: Environment where UAV flies
        :param velocity: the maximum of UAV ' s velocity
        :param max_energy: the maximum of UAV 's battery 's energy
        :param sim_log: record the flying trajectory information
        :param charge_factor: charge UAV ' s factor
        :param time_scale: timescale of simulator
        """
        super().__init__(vid, name, position, env, velocity, sim_log, time_scale)
        self.battery = EnergyComponent("battery", max_energy)
        self.charging_velocity = charging_velocity
        self.max_energy = max_energy
        # self.set_components({"battery": EnergyComponent("battery")})
        self.energy_factor = 0.1
        self.state = self.get_state()
        self.action_space_size = 9
        self.count_time = 0

    def charge(self, charge_time: float):
        """
        charge Uav and charge_time is maximum charge time
        :param charge_time: the charge time
        :return:None
        """
        self.moving_event_count += 1
        self.state = "charging"
        charge_event = MovingSphereEvent(
            position,
            self.moving_event_count,
            self.name,
            self.env,
            self.state,
            self.time_scale)
        try:
            while True:
                charge_event.move()
                charge_power = self.calculate_charge_energy(
                    charge_event.max_moving_time)
                self.battery.add(charge_power)
                if self.battery.is_full():
                    print("charging is over")
                    break
        except simpy.Interrupt as reason:
            print(reason)

    def calculate_charge_energy(self, charge_time):
        charge_power = self.charging_velocity * charge_time
        return charge_power

    def update_position(self, state, init_dist, destination: Point):
        new_position = Point()
        dist = init_dist
        if state in ('charging', 'hovering'):
            new_position = destination
        elif state == 'flying':
            dist = init_dist - self.time_scale * self.velocity
            new_position.x = position.x + self.time_scale * self.velocity
            new_position.y = position.y + self.time_scale * self.velocity
            new_position.z = position.z + self.time_scale * self.velocity
        self.position = new_position
        return dist

    def log_data(self, state):
        log_dict = {
            "name": self.name,
            "state": state,
            "position": [self.position.x, self.position.y, self.position.z],
            "current_energy": self.battery.current_energy
        }
        self.log.write(log_dict)

    def update_energy(self, state):
        if state == "flying":
            self.battery.consume(
                self.time_scale *
                self.velocity *
                self.energy_factor)
        elif state == "hovering":
            self.battery.consume(self.time_scale * self.energy_factor / 2)
        elif state == "charging":
            return self.battery.add(self.charge_factor * self.time_scale)

    def fly_to(self, destination: Point) -> None:
        """
        fly to the destination
        :param destination: fly to the destination
        :param pos: the destination
        :return: None
        """
        moving_event = MovingSphereEvent(
            self.position,
            self.moving_event_count,
            self.name,
            self.env,
            self.state,
            self.time_scale)
        self.moving_event_count += 1
        init_dist = distance(self.position, destination)
        self.env.process(self.run_to(destination, init_dist, "flying"))
        self.env.process(moving_event.move())

    def hover(self, hover_time) -> None:
        """
        hover in the position
        :param hover_time: hover time
        :return: None
        """
        self.state = "hovering"
        hovering_event = MovingSphereEvent(
            self.position,
            self.moving_event_count,
            self.name,
            self.env,
            self.state,
            self.time_scale)
        try:
            while True:
                hovering_event.move()
                if self.has_finished():
                    print("hovering is over")
                    break
        except simpy.Interrupt as reason:
            print(reason)

    def get_state(self):
        state = [
            self.name,
            self.id,
            self.position.x,
            self.position.y,
            self.position.z,
            self.battery.max_energy,
            self.battery.current_energy,
            self.charging_velocity,
            self.velocity,
            self.time_scale]
        return state

    def has_finished(self):
        pass

    def send(self):
        pass

    def receive(self):
        pass


class Robot(Vehicle, ABC):

    def __init__(self, vid, name, position, env: simpy.Environment(),
                 velocity: float, sim_log, timescale):
        super().__init__(vid, name, position, env, velocity, sim_log, timescale)
        self.count_time = 0

    @property
    def get_state(self):
        state = [self.name, self.id,self.position.x, self.position.y, self.position.z]
        return state

    def send(self):
        pass

    def receive(self):
        pass

    def log_data(self, state) -> None:
        pass

    def update_energy(self, state):
        pass



if __name__ == "__main__":
    env1 = simpy.Environment()
    position = Point()
    velocity = 1
    max_energy = 11
    sim_log = "uav"
    charging_velocity = 1
    time_scale = 1
    battery = EnergyComponent("battery", max_energy)
    print(battery.current_energy)
    uav = Uav(1, "uav", position, env1, velocity, 11, "uav", charging_velocity, time_scale)
    print(uav.max_energy)
    print(uav.battery.current_energy)
    uav.battery.add(10)
    print(uav.battery.current_energy)
    print(uav.battery.consume(10))
    print(uav.battery.current_energy)

    print(uav.position.x, uav.position.y, uav.position.z)
    destination = Point(10, 10, 0)
    print(uav.state)
    # uav.fly_to(destination)
    # env1.run(until=30)
    # print(env1.now)
    # print(r'hello')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/26 21:14
# @Author  : S.G.D.
# @File    : event
"""
We think every action is an event, when it runs, event runs. For example, When uav flies from the depot,
we give it an event named moving event. The uav will change its position, battery's energy, and distance
between its position and destination's position until the moving event. We not only give event time property,but
space property. It can be attached to an object.
"""
import simpy
from matplotlib import pyplot as plt

from map.geometry import Point
from map.geometry import distance


class Event:
    """ This is the base class of event. """

    def __init__(self, position: Point, uid, name, style):
        self.position = position
        self.uid = uid
        self.name = name
        self.style = style


class SphereEvent(Event):
    """ This class defines an event whose coverage area is a sphere. """

    def __init__(self, position: Point, uid, name, env, style):
        super().__init__(position, uid, name, style)
        self.radius = 10.0
        self.env = env

    def is_including(self, point):
        """
        Judge sphereEvent whether includes point
        :param point: (x,y,z)
        :return:True or False
        """

        if self.radius * self.radius >= distance(point, self.position)**2:
            return True
        else:
            return False

    def print_running_time(self):

        while True:
            yield self.env.timeout(1)
            print(self.env.now)


class MovingSphereEvent(SphereEvent):
    """This class is subclass of SphereEvent to define SphereEvent moving"""

    def __init__(self, position: Point, uid, name, env, style, max_moving_time):
        """

        :param position: initial position of moving event
        :param uid: the uid of moving event
        :param name:the name of moving event
        :param env: the environment moving event stay
        :param style:the style of moving event
        :param max_moving_time: the max moving time at one time of moving event
        """
        super().__init__(position, uid, name, env, style)
        self.start_time = 0
        self.max_moving_time = max_moving_time
        self.moving = self.env.process(self.move())

    def set_position(self, position: dict):
        self.position = position

    def move(self):
        """
        move the event
        :return:None
        """
        try:
            self.start_time = self.env.now
            print('%s is moving' % self.name)
            yield self.env.timeout(self.max_moving_time)
            print('%s has moved to the end' % self.name)
            self.position.x += 1
            self.position.y += 1
            print(self.position.x,self.position.y)
        except simpy.Interrupt as i:
            self.max_moving_time = self.max_moving_time - (self.env.now - self.start_time)
            print('The event stops because %s' % i.cause)

    def continue_moving(self):
        """
        continue moving the event

        :return:
        """
        self.moving = self.env.process(self.move())

    def stop(self, reason):
        """
        stop the moving event
        :param reason: the reason why the moving event stops
        :return:None
        """
        yield self.env.timeout(5)
        self.moving.interrupt(reason)

class FixedSphereEvent(SphereEvent):
    """This class is subclass of SphereEvent to define SphereEvent fixed"""

    def __init__(self, position, duration_time):
        super().__init__(position)
        self.duration_time = duration_time
        self.fixed()

    def run(self):
        try:
            yield self.env.timeout(self.duration_time)
        except simpy.Interrupt as i:
            print(i)

    def fixed(self):
        """
        use circulation to fix event except interrupted
        :return: sNone
        """
        try:
            while True:
                yield self.env.timeout(self.duration_time)
        except simpy.Interrupt as i:
            print(i)

    def stop(self, reason):
        yield simpy.Interrupt(reason)



if __name__ == "__main__":

    env1 = simpy.Environment()
    position = Point()
    velocity = 1
    max_energy = 11
    #sim_log = "uav"
    charge_factor = 1
    time_scale = 1
    #uav = Uav(1, "uav", position, env1, velocity, 11, "uav", charge_factor, time_scale)
    #print(uav.max_energy)
    moving_event = MovingSphereEvent(position, 0, "moving", env1, "uav", time_scale)
    moving_event.move()
    env1.run(until=5)



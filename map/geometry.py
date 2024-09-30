#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/26 21:14
# @Author  : S.G.D.
# @File    : geometry
import random
from math import sqrt


class Point:
    """ This class is a 3D point by default. """

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x, self.y, self.z = x, y, z

    # def __hash__(self):
    #     return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def distance_to(self, other) -> float:
        if self is other:
            return 0
        else:
            dx, dy, dz = self.x - other.x, self.y - other.y, self.z - other.z
            return sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def vector(self, other):
        "generate vector self point -> other point"
        v = [other.x - self.x, other.y - self.y, other.z - self.z]
        return v


def distance(pa: Point, pb: Point) -> float:
    """
    Return the distance between two points.
    :param pa:
    :param pb:
    :return:
    """
    return pa.distance_to(pb)


class Router:
    """This class is to define the Router"""

    pass


class Facility(Point):
    """This class is to define the subclass of point as point of facility"""

    def __init__(self, x, y, z):
        super().__init__(x, y, z)


def generate_random_time(random_kind="simply-random"):
    if random_kind == "simply-random":
        random_time = random.randint(2, 4)
    return random_time


class InterestingPoint(Point):
    """define the PoI"""

    def __init__(
            self,
            id,
            num,
            point,
            name="interesting-point",
            min_visited_time= 4,
            random_kind="simply-random"):
        """

        :param id:  kind number of point
        :param num: sequence number
        :param point: position
        :param name: kind name
        :param min_visited_time: min visited time
        :param random_kind: the way to generate point
        """
        super(InterestingPoint, self).__init__(point.x, point.y, point.z)
        self.id = id
        self.num = num
        self.name = name
        self.is_visited = False
        self.min_visited_time = min_visited_time
        self.visited_time = 0
        self.random_time = generate_random_time(random_kind)
        self.max_visited_time = self.min_visited_time + self.random_time


class Depot(InterestingPoint):
    "define the starting point"

    def __init__(self, id, num, point, name="depot"):
        super(Depot, self).__init__(id, num, point, name)


class Plane(Point):
    """The class is to define the plane which has width and length"""

    def __init__(
            self,
            plane_id,
            num,
            name,
            center_point: Point,
            length,
            width):
        self.id = plane_id
        self.num = num
        self.name = name
        self.x = center_point.x
        self.y = center_point.y
        self.z = center_point.z
        self.centre_point = center_point
        self.width = width
        self.length = length

    def is_including(self, point: Point):
        """
        to know if the point is in the plane
        :param point: need to be judged whether it is included
        :return: True or False
        """
        if (self.x + self.length / 2) >= point.x >= (self.x - self.length / 2) \
                and (self.y + self.width / 2) >= point.y >= (self.y - self.width / 2):
            return True
        else:
            return False


class ChargingTable(Plane):
    """The class is used as a charging table the uav or others will be charged """

    def __init__(self, tid, num, name, center_point, length, width):
        super().__init__(tid, num, name, center_point, length, width)
        self.visited_time = 0


class Bound:
    def __init__(self):
        pass


class Map:
    def __init__(self, size=[6, 6], center_point=Point(0, 0, 0)):
        self.size = size
        self.center_point = center_point

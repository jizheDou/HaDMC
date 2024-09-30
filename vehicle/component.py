#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/26 21:15
# @Author  : S.G.D.
# @File    : component
from enum import Enum


class ComponentState(Enum):
    """ This class defines the binary state of component. """
    ON = 1
    OFF = 0


class Component:
    """ This class defines the base classes that are vehicle's components other than its controller. """

    def __init__(self, name: str):
        self.name = name
        self.state = ComponentState.ON

    def turn_on(self): self.state = ComponentState.ON

    def turn_off(self): self.state = ComponentState.OFF


class EnergyComponent(Component):
    """ This class defines vehicle's operation on energy. """

    def __init__(self, name: str, max_energy: float = 100):
        super().__init__(name)
        self.max_energy = max_energy
        self.current_energy = self.max_energy

    def add(self, add_energy):
        self.current_energy += add_energy
        self.current_energy = min(self.max_energy, self.current_energy)

    def consume(self, consumed_energy):
        self.current_energy -= consumed_energy

    def is_full(self):
        if self.current_energy >= self.max_energy:
            return True
        else:
            return False

    def is_empty(self):
        if self.current_energy <= 0:
            return True
        else:
            return False


class NetworkComponent(Component):
    pass


if __name__ == "__main__":
    battery = EnergyComponent("battery", 110)
#    print(battery.energy)
    battery.add(100)
    print(battery.energy)
    battery.consume(100)
    print(battery.energy)
    print("hello")

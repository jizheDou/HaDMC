#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/10/26 22:20
# @Author  : S.G.D.
# @File    : logger
from abc import ABCMeta, abstractmethod


class Logger(metaclass=ABCMeta):
    """ This class defines a logger that records what happens in simulation. """

    def __init__(self, log_file_name: str):
        """
        Construct a logger.
        :param log_file_name:
        """
        try:
            self.log = open(log_file_name, mode='a+', encoding='utf-8')
        except OSError as e:
            print(f"Cannot open {log_file_name}!")
            raise FileNotFoundError from e

    @abstractmethod
    def write(self, tag: str, message: str): pass

    def close(self): self.log.close()


class SimLogger(Logger):
    """ This logger is designed for simulator. """

    def __init__(self, log_file_name: str):
        super().__init__(log_file_name)

    def write(self, tag: str, message: str):
        """
        Write into log the given information related to simulation.
        :param tag:
        :param message:
        :return:
        """
        self.log.write(tag + " " + message + '\n')
        self.log.flush()

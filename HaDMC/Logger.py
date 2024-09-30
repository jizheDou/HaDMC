# ！/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/8/20 16:53
# @Author  : doujizhe
# @File    : Logger
# @Software: PyCharm
import logging


class Logger:
    def __init__(self, LEVEL, log_file=None):
        head = '[%(asctime)-15s] [%(levelname)s] %(message)s'
        if LEVEL == 'info':
            logging.basicConfig(level=logging.INFO, format=head)
        elif LEVEL == 'debug':
            logging.basicConfig(level=logging.DEBUG, format=head)
        self.logger = logging.getLogger("mylogger")
        self.logger.setLevel(logging.INFO)
        if log_file is not None:
            fh = logging.FileHandler(log_file)
            self.logger.addHandler(fh)
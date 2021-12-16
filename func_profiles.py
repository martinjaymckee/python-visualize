# import math
import random

import matplotlib.pyplot as plt
import numpy as np


class TriangleProfile:
    def __init__(self, x1=None, y0=None, y1=None, y2=None, y_mults=None, cyclic=False):
        self.__x1 = self.__process_point_defaults(x1, 0.3, 0.7)
        self.__y1 = self.__process_point_defaults(y1, 0.25, 1)
        y_range = (0.3*self.__y1, 0.5*self.__y1)
        if y_mults is not None:
            y_range = (y_mults[0]*self.__y1, y_mults[1]*self.__y1)
        self.__y0 = self.__process_point_defaults(y0, *y_range)
        self.__y2 = self.__y0 if cyclic else self.__process_point_defaults(y2, *y_range)
        self.__cyclic = cyclic
        # print('y0 = {}, y1 = {}, y2 = {}, x1 = {}'.format(self.__y0, self.__y1, self.__y2, self.__x1))

    def __call__(self, x):
        if x <= 0:
            return self.__y0
        elif x >= 1:
            return self.__y2

        upper = x >= self.__x1
        x0, x1 = (self.__x1, 1) if upper else (0, self.__x1)
        y0, y1 = (self.__y1, self.__y2) if upper else (self.__y0, self.__y1)
        m = (y1 - y0) / (x1 - x0)
        b = y1 - (m * x1)
        return (m*x) + b

    def __process_point_defaults(self, v, v0, v1):
        if v is None:
            v = random.uniform(v0, v1)
        elif isinstance(v, tuple):
            v = random.uniform(*v)
        return v


if __name__ == '__main__':
    N = 5
    fig, ax = plt.subplots(1, figsize=(16, 9), constrained_layout=True)
    for _ in range(N):
        xs = np.linspace(0, 1, 100)
        tri = TriangleProfile(x1=(0.6, 0.7), y1=1)
        ys = np.array([tri(x) for x in xs])
        ax.plot(xs, ys)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    plt.show()

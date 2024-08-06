import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np


class CPX:

    def __init__(self, cpxsize: int, alpha: float):

        self.cpxsize = cpxsize
        self.alpha = alpha
        self.mean = np.zeros(cpxsize)
        self.first_step = True

    def __call__(self, dfts):

        y = np.copy(dfts[..., :self.cpxsize])

        # TODO ISSUE #100
        # mean = np.full(y.shape[-1], y[..., 0, :])
        if(self.first_step):
            self.mean = y[..., 0, :].copy()
            self.first_step = False

        for i in range(y.shape[-2]):
            self.mean = np.absolute(y[..., i, :]) * (1 - self.alpha) + self.mean * self.alpha # orig: norm
            if np.abs(self.mean.any())<1e-10:
                self.mean = 1e-10
            y[..., i, :] /= np.sqrt(self.mean)

        return y

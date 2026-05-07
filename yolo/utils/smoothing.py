import numpy as np
import math
import time

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = x0
        self.dx_prev = np.zeros_like(x0)
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class CentroidSmoother:
    def __init__(self, cfg_min_cutoff=0.5, cfg_beta=0.05):
        self.filters = {} 
        self.cfg_min_cutoff = cfg_min_cutoff
        self.cfg_beta = cfg_beta
        self.start_time = time.time()
        
    def reset(self):
        self.filters = {}
        self.start_time = time.time()

    def update(self, cls_id, cx, cy):
        t = time.time() - self.start_time
        curr = np.array([cx, cy], dtype=np.float32)
        
        if cls_id not in self.filters:
            self.filters[cls_id] = OneEuroFilter(t, curr, self.cfg_min_cutoff, self.cfg_beta)
            return cx, cy
        
        res = self.filters[cls_id](t, curr)
        return res[0], res[1]

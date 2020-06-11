import numpy as np
from sklearn.linear_model import LinearRegression

class SlopeDetectionMethod:
    def __init__(self, slope_crit):
        self.name = 'sdm'
        self.data = list()              # collection of batch data
        self.i = 0                      # index at current unchecked batch
        self.slope_crit = slope_crit    # critical slope value: if estimated slope < critical slope, batch is in steady-state
        self.size = 0                   # no. inserted batches
        self.ss_start_point = -1        # detected steady-state starting point
        self.slope = list()             # list: slope at batch i
    
    def insert(self, batch):
        """ add & store new batch of data """
        self.data.append(batch)
        self.size += 1
    
    def steady_state_start_point(self):
        """ return earliest detected steady-state starting point """
        self.update()
        if self.ss_start_point != -1:
            return self.ss_start_point
        # find the earliest detected steady-state starting point
        n = len(self.data[0]) if self.size > 0 else 0
        for i in range(self.size):     # check until the last batch
            if abs(self.slope[i]) < self.slope_crit:
                self.ss_start_point = n * i
                break
        return self.ss_start_point

    def update(self):
        """ check state for every unchecked batch """
        while self.i < self.size:
            model = LinearRegression()
            t = np.arange(len(self.data[self.i]))
            model.fit(t[:, np.newaxis], self.data[self.i])
            slope_deg = np.arctan2(model.coef_[0], 1)  # slope in degree
            self.slope.append(slope_deg)
            self.i += 1                           # move to next batch
        return
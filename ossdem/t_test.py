import numpy as np

class TTest:
    def __init__(self, T_crit):
        self.name = 't_test'
        self.data = list()          # collection of data batches
        self.i = 0                  # index at current unchecked batch
        self.size = 0               # no. received batches
        self.T_crit = T_crit        # critical T value: if estimated T < critical T, batch is steady-state, else transient
        self.T = list()             # list: T-value at batch i
        self.ss_start_point = -1    # detected steady-state starting point
    
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
        n = len(self.data[0]) if self.T else 0
        for i in range(len(self.T)):     # check until the second last batch
            if self.T[i] < self.T_crit:
                self.ss_start_point = n * i
                break
        return self.ss_start_point
    
    def update(self):
        """ update state for second last batch """
        if self.size < 2:   # trivial: max 1 batch received
            return
        n = len(self.data[0])               # batch size
        while self.i < self.size - 1:       # run till second last batch
            x1 = np.mean(self.data[self.i])   # mean_x_i
            x2 = np.mean(self.data[self.i+1]) # mean_x_i+1
            s1 = np.var(self.data[self.i])    # var_s_i
            s2 = np.var(self.data[self.i+1])  # var_s_i+1
            T_i = np.sqrt(n) * abs(x2 - x1) / np.sqrt(s1 + s2)
            self.T.append(T_i)
            self.i += 1                       # move to next batch
        return
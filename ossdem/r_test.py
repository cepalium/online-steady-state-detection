import numpy as np

class RTest:
    def __init__(self, lambda1, lambda2, lambda3, R_crit):
        #
        self.name = 'r_test'
        self.data = list()            # collection of data batches
        self.X = list()               # series of received data
        self.i = 0                    # index at current analyzed data
        self.size = 0
        #
        self.lambda1 = lambda1        # lambda 1 for calculating X_fi
        self.lambda2 = lambda2        # lambda 2 for calculating v_fi
        self.lambda3 = lambda3        # lambda 3 for calculating delta_fi
        self.R_crit = R_crit          # critical R: if estimated R < critical R, batch is steady-state
        #
        self.X_f = list()             # list: X_fi, i=0...n
        self.v_f = list()             # list: v_fi ** 2, i=0...n
        self.delta_f = list()         # list: delta_fi ** 2, i=0...n
        #
        self.R = list()               # list: R_i, i=0...n
        #
        self.ss_start_point = -1      # detected steady-state starting point

    def insert(self, batch):
        """ add & store data batch """
        self.data.append(batch)           # add batch data as sublist
        self.X.extend(batch)              # concatenate new data as items (not sublist)

    def steady_state_start_point(self):
        """ return earliest detected steady-state starting point """
        self.update()
        if self.ss_start_point != -1:
            return self.ss_start_point
        # find the earliest steady-state starting point
        for i in range(len(self.R)):
            if self.R[i] < self.R_crit and i > 50:
                self.ss_start_point = i
                break
        return self.ss_start_point
    
    def update(self):
        """ update state upto latest batch """
        # initialization
        if len(self.X) < 10:
            return
        # initialization
        while self.i < 10:
            if self.i == 0:
                X_f0 = np.mean(self.X[:10])
                X0 = np.mean(self.X[:10])
                v_f0 = np.var(self.X[:10])
                delta_f0 = 2 * np.var(self.X[:10])
                R0 = (2. - self.lambda1) * v_f0 / delta_f0
            self.X_f.append(X_f0)
            self.X[self.i] = X0
            self.v_f.append(v_f0)
            self.delta_f.append(delta_f0)
            self.R.append(self.R_crit)
            self.i += 1
        # update till latest data point
        while self.i < len(self.X):
            X_fi = self.lambda1 * self.X[self.i] + (1. - self.lambda1) * self.X_f[self.i-1]
            v_fi = self.lambda2 * ((self.X[self.i] - self.X_f[self.i-1])**2) + (1. - self.lambda2) * self.v_f[self.i-1]
            delta_fi = self.lambda3 * ((self.X[self.i] - self.X[self.i-1])**2) + (1. - self.lambda3) * self.delta_f[self.i-1]
            R_i = (2. - self.lambda1) * v_fi / delta_fi
            # save to lists for later access
            self.X_f.append(X_fi)
            self.v_f.append(v_fi)
            self.delta_f.append(delta_fi)
            self.R.append(R_i)
            self.i += 1
        return
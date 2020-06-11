import numpy as np
from scipy.stats import norm
from scipy.special import gamma

class ExactBayes:
  def __init__(self, nu=20, gamma=0.2, beta_0=np.array([[0], [0]]), 
                  Sigma=10000*np.matrix([[1,0],[0,1]]), 
                  p=0.2, s_0=0.003, m=20, alpha=0.9):
    self.name = 'eb'
    # parameters
    self.nu = nu
    self.gamma = gamma
    self.beta_0 = beta_0
    self.Sigma = Sigma
    self.p = p
    self.s_0 = s_0
    self.m = m
    self.alpha = alpha
    # 
    self.P_t = [0]    #
    self.P_lcp = [1]  #
    #
    self.data = list()            # collection of data batches
    self.y = list()               # series of received data
    self.t = 1                    # index at current analyzed data
    self.size = 0
    self.ss_start_point = -1      # detected steady-state starting point
    self.ignored_interval = 100

  # public methods
  def insert(self, batch):
    """ add & store data batch """
    self.data.append(batch)           # add batch data as sublist
    self.y.extend(batch)              # concatenate new data as items (not sublist)
    self.size += len(batch)
  
  def steady_state_start_point(self):
    """ return earliest detected steady-state starting point """
    self._update()
    if self.ss_start_point != -1:
      return self.ss_start_point
    # find the earliest steady-state starting point
    for i in range(self.size):
        if self.P_t[i] > self.alpha:
          self.ss_start_point = i
          break
    return self.ss_start_point
  
  # non-public methods
  def _update(self):
    """ update state upto latest batch """
    while self.t < self.size:
      # weighted sampling without replacement if current t >= m
      if self.t > self.m-1:
        locations = sorted(np.random.choice(list(range(self.t)), size=self.m-1, replace=False, p=self.P_lcp / np.sum(self.P_lcp)))  # m-1 locations from 1...t-1
      else:
        locations = list(range(self.t))
      # 
      sum_selected_weight = sum([self.P_lcp[j] for j in locations])
      P_lcp_t1 = {j: self.P_lcp[j] / sum_selected_weight for j in locations}  # normalized
      P_lcp_t = {j:0. for j in locations + [self.t]}
      # calculate P( tau_t = j | y[1:t]), j < t
      for j in locations:
        P_lcp_t[j] = (1. / (np.pi ** (1./2))) \
                      * np.power((np.linalg.det(self._M_st(j, self.t)) / np.linalg.det(self._M_st(j, self.t-1))), 1./2) \
                      * np.power((self._H_st(j, self.t-1) / self._H_st(j, self.t)), (self.t - j + self.nu - 1.) / 2) \
                      * (1. / np.power(self._H_st(j, self.t), 1./2)) \
                      * (((self.t - j + self.nu + 1) / 2) ** (1./2)) \
                      * (1. - (1. / (4 * (self.t - j + self.nu + 1)))) \
                      * (1 - self.p) \
                      * P_lcp_t1[j]
      # calculate P( tau_t = j | y[1:t]), j = t
      P_lcp_t[self.t] = self._P_st(self.t, self.t) * self.p * sum(P_lcp_t1.values())
      # set P ( tau_t | y[1:t]) = 0 at other locations
      for j in range(len(self.P_lcp)):
          self.P_lcp[j] = P_lcp_t[j].item() if j in locations else 0
      self.P_lcp.append(P_lcp_t[self.t].item())   # P_lcp[t]
      # normalize P( tau_t = j | y[1:t])
      norm_P_lcp_t = {k: v / sum(P_lcp_t.values()) for k, v in P_lcp_t.items()}
      # calculate P_t
      P_t = 0.
      for j in norm_P_lcp_t.keys():
        P_t += (norm.cdf((self.s_0 - self._muy_st(j, self.t)[0]) / (self._K_st(j, self.t)[0,0] * ((self._d_st(j, self.t) / (self._d_st(j, self.t) - 2)) ** (1./2)))) \
                - norm.cdf((-self.s_0 - self._muy_st(j, self.t)[0]) / (self._K_st(j, self.t)[0,0] * ((self._d_st(j, self.t) / (self._d_st(j, self.t) - 2)) ** (1./2))))) \
                * norm_P_lcp_t[j]
      self.P_t.append(P_t.item())
      self.t += 1
    return

  def _P_st(self, s, t):
    """ """
    return (np.pi ** (-(t - s + 1) / 2)) \
            * np.power((np.linalg.det(self._M_st(s, t)) / np.linalg.det(self.Sigma)), 1./2) \
            * ((self.gamma ** (self.nu / 2)) / np.power(self._H_st(s, t), ((t - s + 1 + self.nu) / 2))) \
            * (gamma((t - s + 1 + self.nu) / 2) / gamma(self.nu / 2))  

  def _X_st(self, s, t):
    """ return matrix X_st """
    return np.array([np.arange(s, t+1), np.ones(t-s+1)]).T
  
  def _M_st(self, s, t):
    """ return matrix M_st """
    X_st = self._X_st(s, t)
    return (np.dot(X_st.T, X_st) + self.Sigma.I).I

  def _N_st(self, s, t):
    """ return matrix N_st """
    y = np.array([self.y[s:t+1]]).T
    return np.dot(self.Sigma.I, self.beta_0) + np.dot(self._X_st(s, t).T, y)

  def _H_st(self, s, t):
    """ return scalar H_st """
    y = np.array([self.y[s:t+1]]).T
    N_st = self._N_st(s, t)
    return np.dot(y.T, y) + self.gamma + np.dot(np.dot(self.beta_0.T, self.Sigma.I), self.beta_0) - np.dot(np.dot(N_st.T, self._M_st(s, t)), N_st)

  def _muy_st(self, s, t):
    """ return matrix muy_st """
    return np.dot(self._M_st(s, t), self._N_st(s, t))
  
  def _Sigma_st(self, s, t):
    """ return matrix Sigma_st """
    return self._H_st(s, t).item() * self._M_st(s, t) / self._d_st(s, t)
  
  def _d_st(self, s, t):
    """ return scalar d_st """
    return t - s + self.nu + 1
  
  def _K_st(self, s, t):
    """ return matrix K_st """
    return np.diag(np.diag(self._Sigma_st(s, t)))
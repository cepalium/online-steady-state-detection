import numpy as np
import matplotlib.pyplot as plt
import math

class Signal:
    def __init__(self, bias, bias_params, noise, noise_params, n):
        self.bias = bias              # bias singal
        self.noise = noise            # noise type
        self.n = n                    # no. observations
        self.signal = np.zeros(n)        # values of signal at each time t
        self.noise_cache = np.zeros(n)   # values of noise at each time t
        # get bias signal parameters
        if self.bias == 'step':
            self.h1, self.h2, self.h3 = bias_params[0], bias_params[1], bias_params[2] 
            self.T1, self.T2, self.T3 = bias_params[3], bias_params[4], bias_params[5]
        elif self.bias == 'linear':
            self.h, self.T0 = bias_params[0], bias_params[1]
        elif self.bias == 'quadratic':
            self.h, self.T0 = bias_params[0], bias_params[1]
        elif self.bias == 'exponential':
            self.h, self.T0 = bias_params[0], bias_params[1]
        elif self.bias == 'oscillating':
            self.h, self.T0, self.f = bias_params[0], bias_params[1], bias_params[2]
        else:
            pass
        # get noise parameters
        if self.noise == 'ar0':
            self.mu, self.sigma = noise_params[0], noise_params[1]
        elif self.noise == 'ar1':
            self.mu, self.sigma, self.phi1 = noise_params[0], noise_params[1], noise_params[2]
        elif self.noise == 'ar2':
            self.mu, self.sigma = noise_params[0], noise_params[1]
            self.phi2, self.phi3 = noise_params[2], noise_params[3]
        else:
            pass

    def generate(self):
        """ generate the full signal up to observation #n """
        for t in range(self.n):
            self.signal[t] = self._bias_function(t) + self._noise_function(t)
        return self.signal
    
    def plot_signal(self):
        """ plot the generated signal with its true steady-state starting point """
        plt.plot(self.signal, 'b', linewidth=1, label='observed data')
        if self.bias != 'step':                                               # b/c step function doesn't have SS start point
            plt.axvline(x=self.T0, color='k', label='true steady-state point')  # true SS start point = T0
        plt.ylabel('y')
        plt.xlabel('t')
        plt.legend()
        plt.show()
    
    def to_batch(self, batch_size):
        """ segment the whole signal into in-order batches, each has size of batch_size """
        batches = [self.signal[i*batch_size : (i+1)*batch_size] for i in range((self.n + batch_size - 1) // batch_size)]
        return np.array(batches)
        
    def _bias_function(self, t):
        """ return value of bias function at time t w.r.t bias parameters """
        if self.bias == 'step':
            return self._step(t, self.h1, self.h2, self.h3, self.T1, self.T2, self.T3)
        if self.bias == 'linear':
            return self._linear(t, self.h, self.T0)
        if self.bias == 'quadratic':
            return self._quadratic(t, self.h, self.T0)
        if self.bias == 'exponential':
            return self._exponential(t, self.h, self.T0)
        if self.bias == 'oscillating':
            return self._oscillating(t, self.h, self.T0, self.f)
        return
    
    def _noise_function(self, t):
        """ return value of noise at time t w.r.t noise parameters """
        if self.noise == 'ar0':
            return self._ar0(t, self.mu, self.sigma)
        if self.noise == 'ar1':
            return self._ar1(t, self.mu, self.sigma, self.phi1)
        if self.noise == 'ar2':
            return self._ar2(t, self.mu, self.sigma, self.phi2, self.phi3)
        return
    
    def _step(self, t, h1, h2, h3, T1, T2, T3):
        """ return value of step function """
        if t <= T1:
            return h1
        elif T1 < t <= T2:
            return h2
        else:
            return h3
        
    def _linear(self, t, h, T0):
        """ return value of linear function """
        if t <= T0:
            return 1.0 * t * h / T0
        else:
            return h
    
    def _quadratic(self, t, h, T0):
        """ return value of quadratic function """
        if t <= T0:
            return ( 1.0 - ( (t - T0) / (T0 - 1) ) **2 ) * h
        else:
            return h
        
    def _exponential(self, t, h, T0):
        """ return value of exponential function """
        if t <= T0:
            return ( 1.0 - 10**( (1-t) / (T0-1) ) ) * h
        else:
            return ( 1.0 - 10**( -1 ) ) * h
    
    def _oscillating(self, t, h, T0, f):
        """ return value of oscillating function """
        if t <= T0:
            return 1.0 * h * ( (T0 - t) / (T0 - 1) ) * math.sin(math.pi * t / f)
        else:
            return 0
    
    def _ar0(self, t, mu, sigma):
        """ return value of AR0 noise """
        self.noise_cache[t] = np.random.normal(mu, sigma)
        return self.noise_cache[t]
    
    def _ar1(self, t, mu, sigma, phi1):
        """ return value of AR1 noise """
        if t == 0:
            self.noise_cache[t] = np.random.normal(mu, sigma)
        else:
            self.noise_cache[t] = phi1 * self.noise_cache[t-1] + np.random.normal(mu, sigma)
        return self.noise_cache[t]
    
    def _ar2(self, t, mu, sigma, phi2, phi3):
        """ return value of AR2 noise """
        if t == 0:
            self.noise_cache[t] = np.random.normal(mu, sigma)
        elif t == 1:
            self.noise_cache[t] = phi2 * self.noise_cache[t-1] + np.random.normal(mu, sigma)
        else:
            self.noise_cache[t] = phi3 * self.noise_cache[t-2] + phi2 * self.noise_cache[t-1] + np.random.normal(mu, sigma)
        return self.noise_cache[t]
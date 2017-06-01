import numpy as np
from LSTM import LSTMPolicy
from misc import add_delta

class ParameterServer():
  def __init__(self, env):
    self._policy = LSTMPolicy(env.observation_space.shape, env.action_space.n, 0)
    self.weights = self._policy.get_weights()
    # may need to change this into a model

  def add_delta(self, delta):
    self.weights = add_delta(self.weights, delta)

  def async_add(self, delta):
    pass

  def get_weights(self):
    return self.weights

  def get_policy(self):
    return self._policy

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import ray
import time
import sys

from envs import create_env
from LSTM import LSTMPolicy


@ray.remote
class Runner(object):
  """Actor object to start running simulation on workers.

  The gradient computation is also executed from this object.
  """
  def __init__(self, env_name, actor_id, logdir="results/", start=True):
    env = create_env(env_name)
    num_actions = env.action_space.n
    self.policy = LSTMPolicy(env.observation_space.shape, num_actions,
                             actor_id)
  def get_policy(self):
    return self.policy

def train(num_workers, env_name="PongDeterministic-v3"):
  agents = []
  for i in range(num_workers):
    agents.append(Runner.remote(env_name, i))
    time.sleep(0.3)
  return ray.get([a.get_policy.remote() for a in agents])


# #@profile
# import threading 
# def loop(gradient_list, agents, policy, obs):


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run A3C on Ray")
  parser.add_argument("--runners", default=16, type=int,
                      help="Number of simulation workers")
  parser.add_argument("--environment", default="PongDeterministic-v3",
                      type=str, help="The gym environment to use.")
  parser.add_argument("--redis", default=None,
                      type=str, help="Redis Address")


  args = parser.parse_args()
  runners = args.runners
  env_name = args.environment
  addr = args.redis
  if addr:
    ray.init(redis_address=addr)
  else:
    ray.init(num_cpus=runners)
  train(runners, env_name=env_name)


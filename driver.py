from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import ray
import six.moves.queue as queue
import sys
import tensorflow as tf

from envs import create_env
from runner import RunnerThread, process_rollout
from LSTM import LSTMPolicy


@ray.remote
class Runner(object):
  """Actor object to start running simulation on workers.

  The gradient computation is also executed from this object.
  """
  def __init__(self, env_name, actor_id, logdir="results/", start=True):
    env = create_env(env_name)
    self.id = actor_id
    num_actions = env.action_space.n
    self.policy = LSTMPolicy(env.observation_space.shape, num_actions,
                             actor_id)
    self.runner = RunnerThread(env, self.policy, 20)
    self.env = env
    self.logdir = logdir
    if start:
      self.start()

  def pull_batch_from_queue(self):
    """Take a rollout from the queue of the thread runner."""
    rollout = self.runner.queue.get(timeout=600.0)
    while not rollout.terminal:
      try:
        rollout.extend(self.runner.queue.get_nowait())
      except queue.Empty:
        break
    return rollout

  def start(self):
    summary_writer = tf.summary.FileWriter(
        os.path.join(self.logdir, "agent_%d" % self.id))
    self.summary_writer = summary_writer
    self.runner.start_runner(self.policy.sess, summary_writer)

  def compute_gradient(self, params):
    self.policy.set_weights(params)
    rollout = self.pull_batch_from_queue()
    batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
    gradient = self.policy.get_gradients(batch)
    info = {"id": self.id,
            "size": len(batch.a)}
    return gradient, info


def train(num_workers, env_name="PongDeterministic-v3"):
  env = create_env(env_name)
  policy = LSTMPolicy(env.observation_space.shape, env.action_space.n, 0)
  agents = [Runner.remote(env_name, i) for i in range(num_workers)]
  parameters = policy.get_weights()
  gradient_list = [agent.compute_gradient.remote(parameters)
                   for agent in agents]
  steps = 0
  obs = 0
  timing = []
  import time, numpy as np
  from misc import Profiler
  profiler = Profiler()
  from line_profiler import LineProfiler
  prof = LineProfiler()
  p_loop = prof(loop)
  for i in range(2000):
    gradient_list, policy, obs = p_loop(gradient_list, agents, policy, obs)
  prof.print_stats()
  #with profiler:
  #  for i in range(1000):
  #    gradient_list, policy, obs = loop(gradient_list, agents, policy, obs)
  #import ipdb; ipdb.set_trace()

    
  return policy


#@profile
import threading 
def loop(gradient_list, agents, policy, obs):
  done_id, gradient_list = ray.wait(gradient_list)
  gradient, info = ray.get(done_id)[0]
  policy.async_model_update(gradient)
  # policy.model_update(gradient)
  parameters = policy.get_weights(cached=True)
  obs += info["size"]
  gradient_list.extend(
      [agents[info["id"]].compute_gradient.remote(parameters)])
  return gradient_list, policy, obs


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run A3C on Ray")
  parser.add_argument("--runners", default=16, type=int,
                      help="Number of simulation workers")
  parser.add_argument("--environment", default="PongDeterministic-v3",
                      type=str, help="The gym environment to use.")

  args = parser.parse_args()
  runners = args.runners
  env_name = args.environment

  ray.init(num_cpus=runners)
  train(runners, env_name=env_name)

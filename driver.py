from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import ray
import numpy as np
import time
import six.moves.queue as queue
import sys
import tensorflow as tf
import atomicarray

from envs import create_env
from runner import RunnerThread, process_rollout
from LSTM import LSTMPolicy
from ps import ParameterServer
from misc import parameter_delta
norm = lambda x: np.linalg.norm(x)

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
    timing = []
    timing.append(time.time()); self.policy.set_weights(params)
    timing.append(time.time()); self.local_weights = self.policy.get_weights()
    timing.append(time.time()); rollout = self.pull_batch_from_queue()
    timing.append(time.time()); batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
    timing.append(time.time()); gradient, summinfo = self.policy.get_gradients(batch, summarize=True)
    timing.append(time.time()); self.summary_writer.add_summary(tf.Summary.FromString(summinfo), self.policy.local_steps)
    self.summary_writer.flush()
    info = {"id": self.id,
            "size": len(batch.a),
            "time": timing}
    return gradient, info

  def apply_delta(self, delta):
    flattened_deltas = {k: (w * np.ones(w.shape)).flatten() for k, w in delta.items()} # TEMP
    for k in self.weights:
      assert self.weights[k].dtype == np.float64 # TEMP
      atomicarray.increment(self.weights[k], flattened_deltas[k])
    # print(sorted(["%s: %0.2f" % (k[-5:], norm(w)) for k, w in flattened_deltas.items()]))
    # add norms of all flattened_deltas
        # self.weights[k] += delta[k]

  def start_train(self, params, shapes):
    self.weights = params
    self.shapes = shapes
    timing = []
    for i in range(200):
        cur_timing = []
        cur_timing.append(time.time()); self.local_weights = {k: v.reshape(self.shapes[k]) for k, v in self.weights.items()}
        cur_timing.append(time.time()); gradient, info = self.compute_gradient(self.local_weights)
        cur_timing.extend(info['time']);
        cur_timing.append(time.time()); self.policy.model_update(gradient)
        cur_timing.append(time.time()); new_params = self.policy.get_weights()
        cur_timing.append(time.time()); delta = parameter_delta(new_params, self.local_weights)
        cur_timing.append(time.time()); self.apply_delta(delta)
        cur_timing.append(time.time());  timing.append(cur_timing)
    return timing


def train(num_workers, env_name="PongDeterministic-v3"):
  env = create_env(env_name)
  ps = ParameterServer(env)
  parameters = ps.get_weights()
  shapes = {k:v.shape for k, v in parameters.items()}
  parameters = {k: (w * np.ones(w.shape)).flatten() for k, w in parameters.items()} # TEMP
  # flattened_params = {k: np.array(v.flatten(), copy=True) for k, v in parameters.items()}
  p_id = ray.put(parameters)
  s_id = ray.put(shapes)
  agents = []
  for i in range(num_workers):
    agents.append(Runner.remote(env_name, i))
    time.sleep(0.3)
  delta_list = [agent.start_train.remote(p_id, s_id)
                   for agent in agents]
  data = ray.get(delta_list)
  import pickle
  dump = lambda x: pickle.dump(data, open(x, 'wb'))
  import ipdb; ipdb.set_trace()
  return ps.get_policy()


# #@profile
# import threading 
# def loop(gradient_list, agents, policy, obs):


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run A3C on Ray")
  parser.add_argument("--runners", default=1, type=int,
                      help="Number of simulation workers")
  parser.add_argument("--environment", default="PongDeterministic-v3",
                      type=str, help="The gym environment to use.")

  args = parser.parse_args()
  runners = args.runners
  env_name = args.environment

  ray.init(num_cpus=runners)

  # from line_profiler import LineProfiler
  # prof = LineProfiler()

  # p_train = prof(train)
  # 
  # p_train(runners, env_name=env_name)
  train(runners, env_name=env_name)

  # prof.print_stats()


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
from ps import ParameterServer
from misc import parameter_delta


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

  def get_delta(self, params):
    gradient, info = self.compute_gradient(params)
    self.policy.model_update(gradient)
    new_params = self.policy.get_weights()
    return parameter_delta(new_params, params)


def train(num_workers, env_name="PongDeterministic-v3"):
  env = create_env(env_name)
  ps = ParameterServer(env)
  parameters = ps.get_weights()
  agents = [Runner.remote(env_name, i) for i in range(num_workers)]
  delta_list = [agent.get_delta.remote(parameters)
                   for agent in agents]
  steps = 0
  obs = 0
  timing = []
  for i in range(2000):
    done_id, delta_list = ray.wait(delta_list)
    delta, info = ray.get(done_id)[0]
    ps.add_delta(delta) 
    parameters = ps.weights
    obs += info["size"]
    delta_list.extend(
        [agents[info["id"]].compute_gradient.remote(parameters)])
  return policy


# #@profile
# import threading 
# def loop(gradient_list, agents, policy, obs):


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

  # from misc import Profiler
  # profiler = Profiler()
  from line_profiler import LineProfiler
  prof = LineProfiler()

  p_train = prof(train)
  
  p_train(runners, env_name=env_name)

  prof.print_stats()

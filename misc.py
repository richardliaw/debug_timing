from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import cProfile
import io
import pstats


def timestamp():
  return datetime.now().timestamp()


def time_string():
  return datetime.now().strftime("%Y%m%d_%H_%M_%f")


class Profiler(object):
  def __init__(self, p_thres=50):
    self.pr = cProfile.Profile()
    self.thres = p_thres
    self.counter = 0
    self.temp = {}
    self.enabled = False
    pass

  def __enter__(self):
    self.counter += 1
    if not self.enabled:
        self.enabled = True
        self.pr.enable()

  def __exit__(self, type, value, traceback):
    if self.counter % self.thres == 0:
        self.pr.disable()
        self.enabled = False
        ps = pstats.Stats(self.pr)
        #p_items =  sorted(ps.stats.items(), key=lambda kv: kv[1][2] / kv[1][0])[-20:]
        #p_items = [(k, tuple([i, v[0]]) + v[2:]) for i, (k, v) in enumerate(p_items)]
        #ps.stats = {k: v for k, v in p_items}
        #ps.sort_stats('tottime')
        ps.print_stats()
        import ipdb; ipdb.set_trace()

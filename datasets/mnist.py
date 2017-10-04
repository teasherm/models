import os

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def load(data_dir="data/mnist", one_hot=True):
  base_dir = os.path.dirname(data_dir)
  if not os.path.exists(base_dir):
    os.mkdir(base_dir)
  return read_data_sets(data_dir, one_hot=one_hot)

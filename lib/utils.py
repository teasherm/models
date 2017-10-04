import math


def get_batches_per_epoch(batch_size, num_examples):
  return int(math.ceil(num_examples / batch_size))

import numpy as np

from deep_np import utils


def cross_entropy(logits, y_train):
  m = logits.shape[0]

  prob = utils.softmax(logits)
  log_like = -np.log(prob[range(m), y_train])
  data_loss = np.sum(log_like) / m

  return data_loss


def dcross_entropy(logits, y_train):
  m = logits.shape[0]

  grad_y = utils.softmax(logits)
  grad_y[range(m), y_train] -= 1.
  grad_y /= m

  return grad_y

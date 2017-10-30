import numpy as np

from deep_np import utils


class Adam(object):
  def __init__(self, nn, alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    self.alpha = alpha
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.M = {k: np.zeros_like(v) for k, v in nn.model.items()}
    self.R = {k: np.zeros_like(v) for k, v in nn.model.items()}

  def optimize(self, nn, grad, t, ascend=False):
    for k in grad:
      self.M[k] = utils.exp_running_avg(self.M[k], grad[k], self.beta1)
      self.R[k] = utils.exp_running_avg(self.R[k], grad[k]**2, self.beta2)

      m_k_hat = self.M[k] / (1. - self.beta1**(t))
      r_k_hat = self.R[k] / (1. - self.beta2**(t))

      if ascend:
        nn.model[k] += self.alpha * m_k_hat / (np.sqrt(r_k_hat) + self.eps)
        continue
      nn.model[k] -= self.alpha * m_k_hat / (np.sqrt(r_k_hat) + self.eps)

    return nn

import numpy as np

from deep_np import utils


def _pool_forward(X, pool_fun, size=2, stride=2):
  n, d, h, w = X.shape
  h_out = (h - size) // stride + 1
  w_out = (w - size) // stride + 1

  X_reshaped = X.reshape(n * d, 1, h, w)
  X_col = utils.im2col_indices(
      X_reshaped, size, size, padding=0, stride=stride)

  out, pool_cache = pool_fun(X_col)

  out = out.reshape(h_out, w_out, n, d)
  out = out.transpose(2, 3, 0, 1)

  cache = dict(
      X=X, size=size, stride=stride, X_col=X_col, pool_cache=pool_cache)

  return out, cache


def maxpool_forward(X, size=2, stride=2):
  def maxpool(X_col):
    max_idx = np.argmax(X_col, axis=0)
    out = X_col[max_idx, range(max_idx.size)]
    return out, dict(max_idx=max_idx)

  return _pool_forward(X, maxpool, size, stride)


def _pool_backward(dout, dpool_fun, cache):
  X, size, stride, X_col, pool_cache = cache
  n, d, w, h = cache["X"].shape

  dX_col = np.zeros_like(cache["X_col"])
  dout_col = dout.transpose(2, 3, 0, 1).ravel()

  dX = dpool_fun(dX_col, dout_col, cache["pool_cache"])

  dX = utils.col2im_indices(
      dX_col, (n * d, 1, h, w),
      cache["size"],
      cache["size"],
      padding=0,
      stride=cache["stride"])
  dX = dX.reshape(cache["X"].shape)

  return dX


def maxpool_backward(dout, cache):
  def dmaxpool(dX_col, dout_col, pool_cache):
    dX_col[pool_cache["max_idx"], range(dout_col.size)] = dout_col
    return dX_col

  return _pool_backward(dout, dmaxpool, cache)


def sigmoid_forward(X):
  out = utils.sigmoid(X)
  return out, dict(out=out)


def sigmoid_backward(dout, cache):
  return cache["out"] * (1 - cache["out"]) * dout


def tanh_forward(X):
  out = np.tanh(X)
  return out, dict(out=out)


def tanh_backward(dout, cache):
  dX = (1 - cache["out"]**2) * dout
  return dX


def relu_forward(X):
  out = np.maximum(X, 0)
  return out, dict(X=X)


def relu_backward(dout, cache):
  dX = dout.copy()
  dX[cache["X"] <= 0] = 0
  return dX

def dropout_forward(X, p_dropout):
  u = np.random.binomial(1, p_dropout, size=X.shape) / p_dropout
  out = X * u
  return out, dict(u=u)


def dropout_backward(dout, cache):
  return dout * cache["u"]


def bn_forward(X, gamma, beta, cache, momentum=.9, train=True):
  running_mean, running_var = cache

  if train:
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)

    X_norm = (X - mu) / np.sqrt(var + 1e-8)
    out = gamma * X_norm + beta

    cache = dict(X=X, X_norm=X_norm, mu=mu, var=var, gamma=gamma, beta=beta)

    running_mean = utils.exp_running_avg(running_mean, mu, momentum)
    running_var = utils.exp_running_avg(running_var, var, momentum)
  else:
    X_norm = (X - running_mean) / np/sqrt(running_var + 1e-8)
    out = gamma * X_norm + beta
    cache = None

  return out, cache, running_mean, running_var


def bn_backward(dout, cache):
  N, D = cache["X"].shape

  X_mu = cache["X"] = cache["mu"]
  std_inv = 1. / np.sqrt(cache["var"] + 1e-8)

  dX_norm = dout * cache["gamma"]
  dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
  dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

  dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
  dgamma = np.sum(dout * cache["X_norm"], axis=0)
  dbeta = np.sum(dout, axis=0)

  return dX, dgamma, dbeta


def fc_forward(X, W, b):
  out = X @ W + b
  return out, dict(W=W, X=X)


def fc_backward(dout, cache):
  dW = cache["X"].T @ dout
  db = np.sum(dout, axis=0)
  dX = dout @ cache["W"].T

  return dX, dW, db


def conv_forward(X, W, b, stride=1, padding=1):
  """
  N -> # of inputs
  C -> # of channels
  H -> height of input
  W -> width of input
  K -> # of filters
  F -> kernel length (spacial extent)
  S -> stride
  P -> padding

  Output shape -> N, K, (H - F + 2P)/S + 1, (W - F + 2P)/S + 1
  """
  n_filters, d_filter, h_filter, w_filter = W.shape
  n_x, d_x, h_x, w_x = X.shape
  h_out = (h_x - h_filter + 2 * padding) // stride + 1
  w_out = (w_x - w_filter + 2 * padding) // stride + 1

  X_col = utils.im2col_indices(X, h_filter, w_filter)
  W_col = W.reshape(n_filters, -1)

  out = W_col @ X_col + b
  out = out.reshape(n_filters, h_out, w_out, n_x)
  out = out.transpose(3, 0, 1, 2)

  return out, dict(X=X, W=W, b=b, stride=stride, padding=padding, X_col=X_col)


def conv_backward(dout, cache):
  n_filters, d_filter, h_filter, w_filter = cache["W"].shape
  db = np.sum(dout, axis=(0, 2, 3))
  db = db.reshape(n_filters, -1)

  dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filters, -1)
  dW = dout_reshaped @ cache["X_col"].T
  dW = dW.reshape(cache["W"].shape)

  W_reshape = cache["W"].reshape(n_filters, -1)
  dX_col = W_reshape.T @ dout_reshaped
  dX = utils.col2im_indices(
      dX_col,
      cache["X"].shape,
      h_filter,
      w_filter,
      padding=cache["padding"],
      stride=cache["stride"])

  return dX, dW, db

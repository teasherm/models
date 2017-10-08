import numpy as np

from deep_np import utils


def _pool_forward(X, pool_fun, size=2, stride=2):
  n, d, h, w = X.shape
  h_out = (h - size) // stride + 1
  w_out = (w - size) // stride + 1

  X_reshaped = X.reshape(n * d, 1, h, w)
  X_col = utils.im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

  out, pool_cache = pool_fun(X_col)

  out = out.reshape(h_out, w_out, n, d)
  out = out.transpose(2, 3, 0, 1)

  cache = (X, size, stride, X_col, pool_cache)

  return out, cache


def maxpool_forward(X, size=2, stride=2):
  def maxpool(X_col):
    max_idx = np.argmax(X_col, axis=0)
    out = X_col[max_idx, range(max_idx.size)]
    return out, max_idx

  return _pool_forward(X, maxpool, size, stride)


def _pool_backward(dout, dpool_fun, cache):
  X, size, stride, X_col, pool_cache = cache
  n, d, w, h = X.shape

  dX_col = np.zeros_like(X_col)
  dout_col = dout.transpose(2, 3, 0, 1).ravel()

  dX = dpool_fun(dX_col, dout_col, pool_cache)

  dX = utils.col2im_indices(
      dX_col, (n * d, 1, h, w), size, size, padding=0, stride=stride)
  dX = dX.reshape(X.shape)

  return dX


def maxpool_backward(dout, cache):
  def dmaxpool(dX_col, dout_col, pool_cache):
    dX_col[pool_cache, range(dout_col.size)] = dout_col
    return dX_col

  return _pool_backward(dout, dmaxpool, cache)


def relu_forward(X):
  out = np.maximum(X, 0)
  cache = X
  return out, cache


def relu_backward(dout, cache):
  dX = dout.copy()
  dX[cache <= 0] = 0
  return dX


def fc_forward(X, W, b):
  out = X @ W + b
  cache = (W, X)
  return out, cache


def fc_backward(dout, cache):
  W, h = cache

  dW = h.T @ dout
  db = np.sum(dout, axis=0)
  dX = dout @ W.T

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

  cache = (X, W, b, stride, padding, X_col)
  return out, cache


def conv_backward(dout, cache):
  X, W, b, stride, padding, X_col = cache
  n_filters, d_filter, h_filter, w_filter = W.shape
  db = np.sum(dout, axis=(0, 2, 3))
  db = db.reshape(n_filters, -1)

  dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filters, -1)
  dW = dout_reshaped @ X_col.T
  dW = dW.reshape(W.shape)

  W_reshape = W.reshape(n_filters, -1)
  dX_col = W_reshape.T @ dout_reshaped
  dX = utils.col2im_indices(
      dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

  return dX, dW, db

from functools import partial

import numpy as np

from deep_np import layers, losses, utils


def _init_fc_weights(in_dim, out_dim, include_bias=True):
  weights = np.random.randn(in_dim, out_dim) / np.sqrt(in_dim / 2.)
  if include_bias:
    return weights, np.zeros((1, out_dim))
  return weights


def _init_conv_weights(n_filters, n_channels, filter_size, include_bias=True):
  weights = np.random.randn(n_filters, n_channels, filter_size,
                            filter_size) / np.sqrt(n_filters / 2.)
  if include_bias:
    return weights, np.zeros((n_filters, 1))
  return weights


class NeuralNetwork(object):
  def __init__(self, *args, **kwargs):
    pass

  def predict_proba(self, X):
    logits, _ = self.forward(X, train=False)
    return utils.softmax(logits)

  def predict(self, X):
    return np.argmax(self.predict_proba(X), axis=1)

  def train_step(self, X_train, y_train):
    logits, cache = self.forward(X_train)
    loss = losses.cross_entropy(logits, y_train)
    grad = self.backward(logits, y_train, cache)
    return grad, loss

  def forward(self, X, train=True):
    raise NotImplementedError()

  def backward(self, logits, y_train, cache):
    raise NotImplementedError()


class FeedForwardNetwork(NeuralNetwork):
  def __init__(self, input_dim, hidden_dim=128, n_cls=3):
    W1, b1 = _init_fc_weights(input_dim, hidden_dim)
    W2, b2 = _init_fc_weights(hidden_dim, n_cls)
    self.model = dict(W1=W1, b1=b1, W2=W2, b2=b2)

  def forward(self, X, train=True):
    h1, h1_cache = layers.fc_forward(X, self.model["W1"], self.model["b1"])
    h1, nl1_cache = layers.relu_forward(h1)

    logits, logits_cache = layers.fc_forward(h1, self.model["W2"],
                                             self.model["b2"])
    return logits, dict(h1=h1_cache, nl1=nl1_cache, logits=logits_cache)

  # slightly different API to accomodate policy gradient
  def backward(self, grad_y, cache):
    dh1, dW2, db2 = layers.fc_backward(grad_y, cache["logits"])
    dh1 = layers.relu_backward(dh1, cache["nl1"])

    dX, dW1, db1 = layers.fc_backward(dh1, cache["h1"])

    grad = dict(dW1=dW1, db1=db1, dW2=dW2, db2=db2)

    return grad


class ConvolutionalNetwork(NeuralNetwork):
  def __init__(self,
               img_size=28,
               filter_size=3,
               pool_size=2,
               n_channels=1,
               n_filters=10,
               n_cls=10,
               hidden_dim=128):
    super().__init__()
    pool_out_dim = n_filters * img_size // pool_size * img_size // pool_size
    W1, b1 = _init_conv_weights(n_filters, n_channels, filter_size)
    W2, b2 = _init_fc_weights(pool_out_dim, hidden_dim)
    W3, b3 = _init_fc_weights(hidden_dim, n_cls)
    self.model = dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

  def forward(self, X_train, train=False):
    h1, h1_cache = layers.conv_forward(X_train, self.model["W1"],
                                       self.model["b1"])
    h1, nl1_cache = layers.relu_forward(h1)

    hpool, hpool_cache = layers.maxpool_forward(h1)
    h2 = hpool.ravel().reshape(X_train.shape[0], -1)

    h3, h3_cache = layers.fc_forward(h2, self.model["W2"], self.model["b2"])
    h3, nl3_cache = layers.relu_forward(h3)

    logits, logits_cache = layers.fc_forward(h3, self.model["W3"],
                                             self.model["b3"])

    return (logits, dict(
        h1=h1_cache,
        nl1=nl1_cache,
        hpool=hpool_cache,
        hpool_shape=hpool.shape,
        h3=h3_cache,
        nl3=nl3_cache,
        logits=logits_cache))

  def backward(self, logits, y_train, cache):
    grad_y = losses.dcross_entropy(logits, y_train)

    # FC-7
    dh3, dW3, db3 = layers.fc_backward(grad_y, cache["logits"])
    dh3 = layers.relu_backward(dh3, cache["nl3"])

    dh2, dW2, db2 = layers.fc_backward(dh3, cache["h3"])
    dh2 = dh2.ravel().reshape(cache["hpool_shape"])

    # Pool-1
    dpool = layers.maxpool_backward(dh2, cache["hpool"])

    # Conv-1
    dh1 = layers.relu_backward(dpool, cache["nl1"])
    dX, dW1, db1 = layers.conv_backward(dh1, cache["h1"])

    grad = dict(W1=dW1, W2=dW2, W3=dW3, b1=db1, b2=db2, b3=db3)

    return grad


class RecurrentNetwork(NeuralNetwork):
  def __init__(self, vocab_size, char2idx, idx2char, hidden_dim=128):
    self.vocab_size = vocab_size
    self.char2idx = char2idx
    self.idx2char = idx2char
    self.hidden_dim = hidden_dim
    Wxh = _init_fc_weights(vocab_size, hidden_dim, include_bias=False)
    Whh, bh = _init_fc_weights(hidden_dim, hidden_dim)
    Why, by = _init_fc_weights(hidden_dim, vocab_size)
    self.model = dict(Wxh=Wxh, Whh=Whh, bh=bh, Why=Why, by=by)

  def init_hidden_state(self):
    return np.zeros((1, self.hidden_dim))

  def forward(self, X, h, train=True):
    Wxh, Whh, Why = self.model['Wxh'], self.model['Whh'], self.model['Why']
    bh, by = self.model['bh'], self.model['by']

    X_one_hot = np.zeros(self.vocab_size)
    X_one_hot[X] = 1.
    X_one_hot = X_one_hot.reshape(1, -1)

    hprev = h.copy()

    h, h_cache = layers.tanh_forward(X_one_hot @ Wxh + hprev @ Whh + bh)
    y, y_cache = layers.fc_forward(h, Why, by)

    cache = (X_one_hot, Whh, h, hprev, y, h_cache, y_cache)

    if not train:
      y = utils.softmax(y)

    return y, h, cache

  def backward(self, logits, y_train, dh_next, cache):
    X, Whh, h, hprev, y, h_cache, y_cache = cache

    dy = losses.dcross_entropy(logits, y_train)

    dh, dWhy, dby = layers.fc_backward(dy, y_cache)
    dh += dh_next
    dby = dby.reshape((1, -1))

    dh = layers.tanh_backward(dh, h_cache)

    dbh = dh
    dWhh = hprev.T @ dh
    dWxh = X.T @ dh
    dh_next = dh @ Whh.T

    grad = dict(Wxh=dWxh, Whh=dWhh, Why=dWhy, bh=dbh, by=dby)

    return grad, dh_next

  def train_step(self, X_batch, y_batch, h):
    logits_batch = []
    caches = []
    loss = 0.

    for x, y in zip(X_batch, y_batch):
      logits, h, cache = self.forward(x, h, train=True)
      loss += losses.cross_entropy(logits, y)
      logits_batch.append(logits)
      caches.append(cache)

    loss /= X_batch.shape[0]

    dh_next = self.init_hidden_state()
    grads = {k: np.zeros_like(v) for k, v in self.model.items()}

    for t in reversed(range(len(X_batch))):
      grad, dh_next = self.backward(logits_batch[t], y_batch[t], dh_next,
                                    caches[t])

      for k in grads.keys():
        grads[k] += grad[k]

    for k, v in grads.items():
      grads[k] = np.clip(v, 5., -5.)

    return grads, loss, h

  def sample(self, X_seed, h, size=100):
    chars = [self.idx2char[X_seed]]
    idx_list = list(range(self.vocab_size))
    X = X_seed

    for _ in range(size - 1):
      prob, h, _ = self.forward(X, h, train=False)
      idx = np.random.choice(idx_list, p=prob.ravel())
      chars.append(self.idx2char[idx])
      X = idx

    return ''.join(chars)

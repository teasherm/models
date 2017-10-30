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


# Network for pong policy gradient
class PongNetwork(NeuralNetwork):
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

    grad = dict(W1=dW1, b1=db1, W2=dW2, b2=db2)

    return grad


class FeedForwardNetwork(NeuralNetwork):
  def __init__(self, in_dim=784, hidden_dim=128, p_dropout=0.7, n_cls=10):
    self.p_dropout = p_dropout
    W1, b1 = _init_fc_weights(in_dim, hidden_dim)
    beta1 = np.ones((1, hidden_dim))
    gamma1 = np.ones((1, hidden_dim))
    W2, b2 = _init_fc_weights(hidden_dim, hidden_dim)
    beta2 = np.ones((1, hidden_dim))
    gamma2 = np.ones((1, hidden_dim))
    W3, b3 = _init_fc_weights(hidden_dim, n_cls)
    self.model = dict(
        W1=W1,
        b1=b1,
        beta1=beta1,
        gamma1=gamma1,
        W2=W2,
        b2=b2,
        beta2=beta2,
        gamma2=gamma2,
        W3=W3,
        b3=b3)
    self.bn_caches = dict(
        b1_mean=np.zeros((1, H)),
        b1_var=np.zeros((1, H)),
        b2_mean=np.zeros((1, H)),
        b2_var=np.zeros((1, H)))

  def forward(self, X, train=True):
    gamma1, gamma2 = self.model["gamma1"], self.model["gamma2"]
    beta1, beta2 = self.model["beta1"], self.model["beta2"]

    u1, u2 = None, None
    bn1_cache, bn2_cache = None, None

    h1, h1_cache = layers.fc_forward(X, self.model["W1"], self.model["b1"])
    bn1_cache = (self.bn_caches["bn1_mean"], self.bn_caches["bn1_var"])
    h1, bn1_cache, run_mean, run_var = layers.bn_forward(
        h1, gamma1, beta1, bn1_cache, train=train)
    h1, nl1_cache = layers.relu_forward(h1)

    self.bn_caches["bn1_mean"], self.bn1_mean["bn1_var"] = run_mean, run_var

    if train:
      h1, u1 = layers.dropout_forward(h1, self.p_dropout)

    h2, h2_cache = layers.fc_forward(X, self.model["W2"], self.model["b2"])
    bn2_cache = (self.bn_caches["bn2_mean"], self.bn_caches["bn2_var"])
    h2, bn2_cache, run_mean, run_var = layers.bn_forward(
        h2, gamma2, beta2, bn2_cache, train=train)
    h2, nl2_cache = layers.relu_forward(h2)

    self.bn_caches["bn2_mean"], self.bn2_mean["bn2_var"] = run_mean, run_var

    if train:
      h2, u2 = layers.dropout_forward(h2, self.p_dropout)

    logits, logits_cache = layers.fc_forward(h2, self.model["W3"],
                                             self.model["b3"])

    return logits, dict(
        X=X,
        h1=h1_cache,
        h2=h2_cache,
        logits=logits_cache,
        nl1=nl1_cache,
        nl2=nl2_cache,
        u1=u1,
        u2=u2,
        bn1=bn1_cache,
        b2=bn2_cache)

  def backward(self, logits, y_train, cache):
    grad_y = loss.dcross_entropy(logits, y_train)

    dh2, dW3, db3 = layers.fc_backward(grad_y, cache["logits"])
    dh2 = layers.relu_backward(dh2, cache["nl2"])
    dh2 = layers.dropout_backward(dh2, cache["u2"])
    dh2, dgamma2, dbeta2 = layers.bn_backward(dh2, cache["bn2"])

    dh1, dW2, db2 = layers.fc_backward(dh2, h2_cache)
    dh1 = self.relu_backward(dh1, cache["nl1"])
    dh1 = layers.dropout_backward(dh1, cache["u1"])
    dh1, dgamma1, dbeta1 = layers.bn_backward(dh1, cache["bn1"])

    _, dW1, db1 = layers.fc_backward(dh1, cache["h1"])

    return dict(
        W1=dW1,
        b1=db1,
        W2=dW2,
        b2=db2,
        W3=dW3,
        b3=db3,
        gamma1=dgamma1,
        beta1=dbeta1,
        gamma2=dgamma2,
        beta2=dbeta2)


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


class LSTM(RecurrentNetwork):
  def __init__(self, vocab_size, char2idx, idx2char, hidden_dim=128):
    self.vocab_size = vocab_size
    self.char2idx = char2idx
    self.idx2char = idx2char
    self.hidden_dim = hidden_dim
    Wf, bf = _init_fc_weights(vocab_size + hidden_dim, hidden_dim)
    Wi, bi = _init_fc_weights(vocab_size + hidden_dim, hidden_dim)
    Wc, bc = _init_fc_weights(vocab_size + hidden_dim, hidden_dim)
    Wo, bo = _init_fc_weights(vocab_size + hidden_dim, hidden_dim)
    Wy, by = _init_fc_weights(hidden_dim, vocab_size)
    self.model = dict(
        Wf=Wf, bf=bf, Wi=Wi, bi=bi, Wc=Wc, bc=bc, Wo=Wo, bo=bo, Wy=Wy, by=by)

  def init_hidden_state(self):
    return (np.zeros((1, self.hidden_dim)), np.zeros((1, self.hidden_dim)))

  def forward(self, X, state, train=True):
    h_old, c_old = state

    X_one_hot = np.zeros(self.vocab_size)
    X_one_hot[X] = 1.
    X_one_hot = X_one_hot.reshape(1, -1)

    X = np.column_stack((h_old, X_one_hot))

    hf, hf_cache = layers.fc_forward(X, self.model["Wf"], self.model["bf"])
    hf, hf_sigm_cache = layers.sigmoid_forward(hf)

    hi, hi_cache = layers.fc_forward(X, self.model["Wi"], self.model["bi"])
    hi, hi_sigm_cache = layers.sigmoid_forward(hi)

    ho, ho_cache = layers.fc_forward(X, self.model["Wo"], self.model["bo"])
    ho, ho_sigm_cache = layers.sigmoid_forward(ho)

    hc, hc_cache = layers.fc_forward(X, self.model["Wc"], self.model["bc"])
    hc, hc_tanh_cache = layers.tanh_forward(hc)

    c = hf * c_old + hi * hc
    c, c_tanh_cache = layers.tanh_forward(c)

    h = ho * c

    y, y_cache = layers.fc_forward(h, self.model["Wy"], self.model["by"])

    cache = (X, hf, hi, ho, hc, hf_cache, hf_sigm_cache, hi_cache,
             hi_sigm_cache, ho_cache, ho_sigm_cache, hc_cache, hc_tanh_cache,
             c_old, c, c_tanh_cache, y_cache)

    if not train:
      y = utils.softmax(y)

    return y, (h, c), cache

  def backward(self, logits, y_train, d_next, cache):
    X, hf, hi, ho, hc, hf_cache, hf_sigm_cache, hi_cache, hi_sigm_cache, ho_cache, \
        ho_sigm_cache, hc_cache, hc_tanh_cache, c_old, c, c_tanh_cache, y_cache = cache
    dh_next, dc_next = d_next

    dy = losses.dcross_entropy(logits, y_train)

    dh, dWy, dby = layers.fc_backward(dy, y_cache)
    dh += dh_next

    dho = c * dh
    dho = layers.sigmoid_backward(dho, ho_sigm_cache)

    dc = ho * dh
    dc = layers.tanh_backward(dc, c_tanh_cache)
    dc = dc + dc_next

    dhf = c_old * dc
    dhf = layers.sigmoid_backward(dhf, hf_sigm_cache)

    dhi = hc * dc
    dhi = layers.sigmoid_backward(dhi, hi_sigm_cache)

    dhc = hi * dc
    dhc = layers.tanh_backward(dhc, hc_tanh_cache)

    dXo, dWo, dbo = layers.fc_backward(dho, ho_cache)
    dXc, dWc, dbc = layers.fc_backward(dhc, hc_cache)
    dXi, dWi, dbi = layers.fc_backward(dhi, hi_cache)
    dXf, dWf, dbf = layers.fc_backward(dhf, hf_cache)

    dX = dXo + dXc + dXi + dXf
    dh_next = dX[:, :self.hidden_dim]
    dc_next = hf * dc

    grad = dict(Wf=dWf, Wi=dWi, Wc=dWc, Wo=dWo, Wy=dWy, bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)

    return grad, (dh_next, dc_next)

  def train_step(self, X_batch, y_batch, state):
    logits_batch = []
    caches = []
    loss = 0.

    for x, y_true in zip(X_batch, y_batch):
      logits, state, cache = self.forward(x, state, train=True)
      loss += losses.cross_entropy(logits, y_true)
      logits_batch.append(logits)
      caches.append(cache)

    loss /= X_batch.shape[0]

    # backward
    d_next = self.init_hidden_state()

    grads = {k: np.zeros_like(v) for k, v in self.model.items()}

    for y_pred, y_true, cache in reversed(list(zip(logits_batch, y_batch, caches))):
      grad, d_next = self.backward(y_pred, y_true, d_next, cache)

      for k in grads.keys():
        grads[k] += grad[k]

    for k, v in grads.items():
      grads[k] = np.clip(v, -5., 5.)

    return grads, loss, state

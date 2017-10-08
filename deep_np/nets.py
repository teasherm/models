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

  def forward(self):
    raise NotImplementedError()

  def backward(self):
    raise NotImplementedError()


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

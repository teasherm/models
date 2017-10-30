import numpy as np
from tqdm import tqdm

from datasets.mnist import load as load_mnist
from datasets.shakespeare import load as load_shakespeare
from deep_np import nets, optimizers, reinforcement, utils
from lib.utils import get_batches_per_epoch


def train_cnn(batch_size=256, epochs=50, print_every=10):
  def _prepro(X, mean):
    X = X.transpose(0, 3, 1, 2)
    return X - mean

  mnist = load_mnist(one_hot=False, reshape=False)
  train_mean = np.mean(mnist.train.images)
  batches_per_epoch = get_batches_per_epoch(batch_size,
                                            mnist.train.num_examples)
  nn = nets.ConvolutionalNetwork()
  optimizer = optimizers.Adam(nn)

  for e in tqdm(range(epochs)):
    for b in range(1, batches_per_epoch + 1):
      t = e * batches_per_epoch + b
      X, y = mnist.train.next_batch(batch_size)
      X = _prepro(X, train_mean)
      grad, loss = nn.train_step(X, y)
      nn = optimizer.optimize(nn, grad, t)
      if t % print_every == 0:
        print("Iter-{} loss: {:4f}".format(t, loss))

    X_val, y_val = prepro(mnist.validation.images,
                          train_mean), mnist.validation.labels
    val_acc = utils.accuracy(y_val, nn.predict(X_val))
    print('Epoch-{} loss: {:4f} val_acc: {:4f}'.format(e, loss, val_acc))


def train_rnn(batch_size=256, epochs=50, print_every=100):
  X, y, char2idx, idx2char = load_shakespeare()
  batches_per_epoch = get_batches_per_epoch(batch_size, X.size)
  nn = nets.RecurrentNetwork(len(char2idx), char2idx, idx2char)
  optimizer = optimizers.Adam(nn)
  for e in tqdm(range(epochs)):
    hidden_state = nn.init_hidden_state()
    for b in range(1, batches_per_epoch + 1):
      t = e * batches_per_epoch + b
      grad, loss, h = nn.train_step(X[(b - 1) * batch_size:b * batch_size],
                                    y[(b - 1) * batch_size:b * batch_size],
                                    hidden_state)
      optimizer.optimize(nn, grad, t)
      if t % print_every == 0:
        print(nn.sample(X[0], hidden_state, 100))


if __name__ == "__main__":
  reinforcement.train_policy_gradient()

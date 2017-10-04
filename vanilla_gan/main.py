import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from datasets.mnist import load as load_mnist
from lib.utils import get_batches_per_epoch
from vanilla_gan import model


def sample_z(batch_size, z_dim):
  return np.random.uniform(-1, 1, size=(batch_size, z_dim))


def plot_samples(samples):
  fig = plt.figure(figsize=(4, 4))
  gs = gridspec.GridSpec(4, 4)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

  return fig


def save_fig(i, path="results/vanilla_gan"):
  if not os.path.exists(path):
    os.makedirs(path)
  plt.savefig("{}/{}.png".format(path, str(i).zfill(3)), bbox_inches="tight")
  plt.close()


def train(epochs=50, batch_size=16, z_dim=100):
  mnist = load_mnist()
  batches_per_epoch = get_batches_per_epoch(batch_size,
                                            mnist.train.num_examples)
  graph = tf.Graph()
  with tf.Session(graph=graph) as sess:
    var_dict = model.build_graph(z_dim)
    model.init(sess)
    for e in tqdm(range(epochs)):
      for b in range(batches_per_epoch):
        X, _ = mnist.train.next_batch(batch_size)
        X.reshape([-1, 784])

        z = sample_z(batch_size, z_dim)
        d_loss = model.optimize_discriminator(sess, var_dict, X, z)

        z = sample_z(batch_size, z_dim)
        g_loss = model.optimize_generator(sess, var_dict, z)

      print('Epoch: {}'.format(e))
      print('D loss: {:.4}'.format(d_loss))
      print('G loss: {:.4}'.format(g_loss))
      print()

      z = sample_z(batch_size, z_dim)
      samples = model.sample_from_generator(sess, var_dict, z)
      fig = plot_samples(samples)
      save_fig(e)


if __name__ == "__main__":
  train()

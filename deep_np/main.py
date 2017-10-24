import gym
import numpy as np
from tqdm import tqdm

from datasets.mnist import load as load_mnist
from datasets.shakespeare import load as load_shakespeare
from deep_np import nets, optimizers, utils
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


def train_policy_gradient(render=False, gamma=0.99, batch_size=100):
  def _prepro(X):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    X = X[35:195]  # crop
    X = X[::2, ::2, 0]  # downsample by factor of 2
    X[X == 144] = 0  # erase background (background type 1)
    X[X == 109] = 0  # erase background (background type 2)
    X[X != 0] = 1  # everything else (paddles, ball) just set to 1
    return X.astype(np.float).ravel()

  def _discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
      if r[t] != 0: running_add = 0
      running_add = running_add * gamma + r[t]
      discounted_r[t] = running_add
    return discounted_r

  def _sample_action(aprob):
    # 0: NO-OP
    # 2: DOWN
    # 3: UP
    eps = np.random.uniform()
    if eps < aprob[0]:
      return 0
    elif eps < aprob[0] + aprob[1]:
      return 2
    return 3

  def _y_from_action(action):
    if action == 0:
      return [1., 0., 0.]
    elif action == 2:
      return [0., 1., 0.]
    return [0., 0., 1.]

  env = gym.make("Pong-v0")
  obs = env.reset()
  prev_x = None
  caches, dlogps, drs = {}, [], []
  running_reward = None
  reward_sum = 0
  episode_no = 0

  nn = nets.FeedForwardNetwork(80 * 80, n_cls=3)
  optimizer = optimizers.Adam(nn)
  grad_buffer = {k: np.zeros_like(v) for k, v in nn.model.items()}

  batch_no = 0
  while True:
    if render: env.render()

    cur_x = _prepro(obs)
    x = cur_x - prev_x if prev_x is not None else np.zeros(80 * 80)
    prev_x = cur_x

    logits, cache = nn.forward(x)
    aprob = utils.softmax(logits)[0]
    action = _sample_action(aprob)

    for layer_name, layer_cache in cache.items():
      if layer_name not in caches: caches[layer_name] = {}
      for tensor_name, tensor in layer_cache.items():
        if tensor_name not in caches[layer_name]:
          caches[layer_name][tensor_name] = []
        caches[layer_name][tensor_name].append(tensor)

    y = _y_from_action(action)
    dlogps.append(y - aprob)

    obs, reward, done, info = env.step(action)
    reward_sum += reward
    drs.append(reward)
    if done:
      episode_no += 1

      epdlogp = np.vstack(dlogps)
      epr = np.vstack(drs)

      discounted_epr = _discount_rewards(epr)
      discounted_epr -= np.mean(discounted_epr)
      discounted_epr /= np.std(discounted_epr)

      epdlogp *= discounted_epr

      # stack episode caches for backward pass
      for layer_name, layer_cache in caches.items():
        for tensor_name, tensor_list in layer_cache.items():
          if "W" in tensor_name:
            # take first W, should be the same for all steps!
            assert abs(tensor_list[0].sum() - tensor_list[1].sum()) < 0.1
            caches[layer_name][tensor_name] = tensor_list[0]
            continue
          # stack inputs and activations from each step
          caches[layer_name][tensor_name] = np.vstack(tensor_list) if len(
              tensor_list[0].shape) == 2 else np.stack(tensor_list)

      grad = nn.backward(epdlogp, caches)
      caches, dlogps, drs = {}, [], []
      for k in nn.model:
        grad_buffer[k] += grad["d" + k]

      if episode_no % batch_size == 0:
        print("Completed batch no: {}, optimizing policy...".format(batch_no))
        batch_no += 1
        optimizer.optimize(nn, grad_buffer, batch_no)
        grad_buffer = {k: np.zeros_like(v) for k, v in nn.model.items()}

      running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
      print("Resetting env... Ep reward total: {}. Running mean: {}".format(
          reward_sum, running_reward))
      reward_sum = 0
      obs = env.reset()  # reset env
      prev_x = None


if __name__ == "__main__":
  train_rnn()

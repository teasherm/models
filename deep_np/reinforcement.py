import os

import gym
import numpy as np
import pickle

from deep_np import nets, optimizers, utils


def train_policy_gradient(render=False,
                          gamma=0.99,
                          batch_size=50,
                          resume=False):
  env = gym.make("Pong-v0")
  obs = env.reset()
  prev_x = None
  caches, dlogps, drs = {}, [], []
  running_reward = None
  reward_sum = 0
  episode_no = 0

  nn = _load_obj("model") if resume else nets.PongNetwork(80 * 80, n_cls=3)
  optimizer = _load_obj("optimizer") if resume else optimizers.Adam(nn)
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
    # Inline backprop
    dlogps.append(y - aprob)

    obs, reward, done, info = env.step(action)
    reward_sum += reward
    drs.append(reward)
    if done:
      episode_no += 1

      epdlogp = np.vstack(dlogps)
      epr = np.vstack(drs)

      discounted_epr = _discount_rewards(epr, gamma)
      discounted_epr -= np.mean(discounted_epr)
      discounted_epr /= np.std(discounted_epr)

      epdlogp *= discounted_epr

      # stack episode caches for backward pass
      for layer_name, layer_cache in caches.items():
        for tensor_name, tensor_list in layer_cache.items():
          if "W" in tensor_name:
            # take first W, should be the same for all steps!
            assert np.allclose(tensor_list[0], tensor_list[1])
            caches[layer_name][tensor_name] = tensor_list[0]
            continue
          # stack inputs and activations from each step
          caches[layer_name][tensor_name] = np.vstack(tensor_list) if len(
              tensor_list[0].shape) == 2 else np.stack(tensor_list)

      grad = nn.backward(epdlogp, caches)
      caches, dlogps, drs = {}, [], []
      for k in nn.model:
        grad_buffer[k] += grad[k]

      if episode_no % batch_size == 0:
        print("Completed batch no: {}, optimizing policy...".format(batch_no))
        batch_no += 1
        optimizer.optimize(nn, grad_buffer, batch_no, ascend=True)
        grad_buffer = {k: np.zeros_like(v) for k, v in nn.model.items()}
        _save_obj(nn, "model")
        _save_obj(optimizer, "optimizer")

      running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
      print("Resetting env... Ep reward total: {}. Running mean: {}".format(
          reward_sum, running_reward))
      reward_sum = 0
      obs = env.reset()  # reset env
      prev_x = None


def _prepro(X):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  X = X[35:195]  # crop
  X = X[::2, ::2, 0]  # downsample by factor of 2
  X[X == 144] = 0  # erase background (background type 1)
  X[X == 109] = 0  # erase background (background type 2)
  X[X != 0] = 1  # everything else (paddles, ball) just set to 1
  return X.astype(np.float).ravel()


def _discount_rewards(r, gamma):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


def _sample_action(aprob):
  # 0: NO-OP | 2: UP | 3: DOWN
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


def _save_obj(obj, name, save_path="data/rl"):
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  with open(os.path.join(save_path, "{}.p".format(name)), "wb") as f:
    pickle.dump(obj, f)


def _load_obj(name, save_path="data/rl"):
  with open(os.path.join(save_path, "{}.p".format(name)), "rb") as f:
    return pickle.load(f)

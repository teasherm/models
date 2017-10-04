# Recreation of pytorch tutorial, drawing from OpenAI baselines

import math
import random
from collections import namedtuple
from itertools import count

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from deep_q import graph as deepq_graph

Transition = namedtuple("Transition", ("state", "action", "next_state",
                                       "reward", "done"))


class ReplayMemory(object):
  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0

  def push(self, *args):
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = Transition(*args)
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)


class ExponentialSchedule(object):
  def __init__(self, decay_timesteps, final_p, initial_p=1.):
    self.initial_p = initial_p
    self.final_p = final_p
    self.decay_timesteps = decay_timesteps

  def value(self, t):
    return self.final_p + (self.initial_p - self.final_p) * \
        math.exp(-1. / self.decay_timesteps * t)


def get_cart_location(env, screen_width):
  world_width = env.x_threshold * 2
  scale = screen_width / world_width
  return int(env.state[0] * scale + screen_width / 2.0)


def get_screen(env, screen_width=600, view_width=320):
  screen = env.render(mode="rgb_array")
  screen = screen[160:320, :]
  cart_location = get_cart_location(env, screen_width)
  if cart_location < view_width // 2:
    slice_range = slice(view_width)
  elif cart_location > (screen_width - view_width // 2):
    slice_range = slice(-view_width, None)
  else:
    slice_range = slice(cart_location - view_width // 2,
                        cart_location + view_width // 2)
  screen = screen[:, slice_range, :]
  screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
  return cv2.resize(screen, (80, 40), interpolation=cv2.INTER_CUBIC)


def plot_durations(episode_durations):
  plt.clf()
  plt.title("Training..")
  plt.xlabel("Episode")
  plt.ylabel("Duration")
  plt.plot(np.array(episode_durations))
  plt.pause(0.001)


def train_cartpole(num_episodes=1000,
                   batch_size=128,
                   learning_starts=128,
                   training_freq=1,
                   training_update_freq=50):
  plt.figure()
  env = gym.make("CartPole-v0").unwrapped
  memory = ReplayMemory(10000)
  exploration = ExponentialSchedule(200, 0.05, initial_p=0.9)
  episode_durations, episode_rewards = [], [0.0]
  graph = tf.Graph()
  with tf.Session(graph=graph) as sess:
    var_dict = deepq_graph.build_graph()
    deepq_graph.initialize(sess)
    deepq_graph.sync_target_network(sess, var_dict)
    t = 0
    for i in range(num_episodes):
      env.reset()
      last_screen = get_screen(env)
      current_screen = get_screen(env)
      state = current_screen - last_screen
      for e_t in count():
        update_eps = exploration.value(t)
        action = deepq_graph.select_action(sess, var_dict, state, update_eps)
        t += 1
        obs, reward, done, info = env.step(action)
        last_screen = current_screen
        current_screen = get_screen(env)

        next_state = current_screen - last_screen

        memory.push(state, action, next_state, reward, float(done))
        episode_rewards[-1] += reward
        state = next_state

        if t > learning_starts and t % training_freq == 0:
          transitions = memory.sample(batch_size)
          batch = Transition(*zip(*transitions))

          state_batch = np.array(batch.state)
          action_batch = np.array(batch.action)
          reward_batch = np.array(batch.reward)
          next_state_batch = np.array(batch.next_state)
          done_batch = np.array(batch.done)

          deepq_graph.train(sess, var_dict,
                            (state_batch, action_batch, reward_batch,
                             next_state_batch, done_batch))

        if t > learning_starts and t % training_update_freq == 0:
          deepq_graph.sync_target_network(sess, var_dict)

        if done:
          episode_durations.append(e_t + 1)
          episode_rewards.append(0.0)
          plot_durations(episode_durations)
          break


if __name__ == "__main__":
  train_cartpole()

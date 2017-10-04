import numpy as np
import tensorflow as tf

from lib import ops


def _get_collection(relative_name):
  return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                           tf.get_variable_scope().name + "/" + relative_name)


def _q_func(obs, num_actions, scope, reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    x = ops.conv2d(
        obs,
        16,
        kernel_size=(5, 5),
        strides=(2, 2),
        activation=None,
        name="conv1")
    x = ops.batch_normalization(x, name="bn1")
    x = tf.nn.relu(x)

    x = ops.conv2d(
        x,
        32,
        kernel_size=(5, 5),
        strides=(2, 2),
        activation=None,
        name="conv2")
    x = ops.batch_normalization(x, name="bn2")
    x = tf.nn.relu(x)

    x = ops.conv2d(
        x,
        64,
        kernel_size=(5, 5),
        strides=(2, 2),
        activation=None,
        name="conv3")
    x = ops.batch_normalization(x, name="bn3")
    x = tf.nn.relu(x)

    x = tf.contrib.layers.flatten(x)
    action_values = ops.dense(x, num_actions, activation=None)
    return action_values



def build_graph(num_actions=2, gamma=0.999):
  with tf.variable_scope("deepq"):
    #
    ## select_action subgraph
    #

    # placeholders
    observations_ph = tf.placeholder(
        tf.float32, shape=(None, 40, 80, 3), name="observation")
    stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
    update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")
    eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

    # variables to get actions and update epsilon
    q_values = _q_func(observations_ph, num_actions, scope="q_func")
    deterministic_actions = tf.argmax(q_values, axis=1)

    batch_size = tf.shape(observations_ph)[0]
    random_actions = tf.random_uniform(
        tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
    chose_random = tf.random_uniform(
        tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
    stochastic_actions = tf.where(chose_random, random_actions,
                                  deterministic_actions)

    output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions,
                             lambda: deterministic_actions)

    update_eps_expr = eps.assign(
        tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))

    #
    ## train subgraph (overlaps with select_action_subgraph above)
    #

    # placeholders
    obs_t_ph = tf.placeholder(
        tf.float32, shape=(None, 40, 80, 3), name="obs_t")
    action_t_ph = tf.placeholder(tf.int32, shape=(None), name="action_t")
    reward_t_ph = tf.placeholder(tf.float32, shape=(None), name="reward_t")
    obs_tp1_ph = tf.placeholder(
        tf.float32, shape=(None, 40, 80, 3), name="obs_tp1")
    done_mask_ph = tf.placeholder(tf.float32, shape=(None), name="done_mask")

    # Reuse weights from select_action q_func above by using same
    # scope and setting reuse=True
    q_t = _q_func(obs_t_ph, num_actions, scope="q_func", reuse=True)
    q_func_vars = _get_collection("q_func")

    q_t_selected = tf.reduce_sum(
        q_t * tf.one_hot(action_t_ph, num_actions), axis=1)

    q_tp1 = _q_func(obs_t_ph, num_actions, scope="target_q_func")
    target_q_func_vars = _get_collection("target_q_func")

    q_tp1_best = tf.reduce_max(q_tp1, axis=1)
    q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best
    q_t_selected_target = reward_t_ph + gamma * q_tp1_best_masked

    # stop gradient on target, as only want to train action network
    # which is periodically copied to target network below
    loss = tf.losses.huber_loss(q_t_selected,
                                tf.stop_gradient(q_t_selected_target))
    train_op = tf.train.AdamOptimizer().minimize(loss, var_list=q_func_vars)

    #
    ## update target subgraph
    #

    # operations to copy weights from trained q network to target
    update_target_expr = []
    for var, var_target in zip(
        sorted(q_func_vars, key=lambda v: v.name),
        sorted(target_q_func_vars, key=lambda v: v.name)):
      update_target_expr.append(var_target.assign(var))
    update_target_expr = tf.group(*update_target_expr)

    # return placeholders and tensors for use in functions below
    return dict(
        observations_ph=observations_ph,
        stochastic_ph=stochastic_ph,
        update_eps_ph=update_eps_ph,
        obs_t_ph=obs_t_ph,
        action_t_ph=action_t_ph,
        reward_t_ph=reward_t_ph,
        obs_tp1_ph=obs_tp1_ph,
        done_mask_ph=done_mask_ph,
        act=output_actions,
        train_op=train_op,
        update_target_network=update_target_expr,
        update_eps=update_eps_expr)


def initialize(sess):
  sess.run(tf.global_variables_initializer())


def select_action(sess, var_dict, state, update_eps):
  actions, _ = sess.run(
      [var_dict["act"], var_dict["update_eps"]],
      feed_dict={
          var_dict["observations_ph"]: state[np.newaxis],
          var_dict["stochastic_ph"]: True,
          var_dict["update_eps_ph"]: update_eps
      })
  return actions[0]


def train(sess, var_dict, batch):
  states, actions, rewards, next_states, dones = batch
  sess.run(
      var_dict["train_op"],
      feed_dict={
          var_dict["obs_t_ph"]: states,
          var_dict["action_t_ph"]: actions,
          var_dict["reward_t_ph"]: rewards,
          var_dict["obs_tp1_ph"]: next_states,
          var_dict["done_mask_ph"]: dones,
      })


def sync_target_network(sess, var_dict):
  sess.run(var_dict["update_target_network"])

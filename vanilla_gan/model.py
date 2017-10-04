import tensorflow as tf

from lib.ops import xavier_initializer, sigmoid_loss

relu = tf.nn.relu


def _discriminator(X, reuse=False):
  with tf.variable_scope("discriminator") as scope:
    if reuse: scope.reuse_variables()
    d_l1 = tf.layers.dense(
        X,
        128,
        activation=relu,
        kernel_initializer=xavier_initializer(X),
        name="layer1")
    d_logits = tf.layers.dense(
        d_l1, 1, kernel_initializer=xavier_initializer(d_l1), name="layer2")
    d_prob = tf.nn.sigmoid(d_logits, name="prob")
    return d_prob, d_logits


def _generator(z):
  with tf.variable_scope("generator") as scope:
    g_l1 = tf.layers.dense(
        z,
        128,
        activation=relu,
        kernel_initializer=xavier_initializer(z),
        name="layer1")
    g_l2 = tf.layers.dense(
        g_l1, 784, kernel_initializer=xavier_initializer(g_l1), name="layer2")
    g_prob = tf.nn.sigmoid(g_l2, name="prob")
    return g_prob


def build_graph(z_dim):
  X = tf.placeholder(tf.float32, shape=[None, 784], name="X")
  z = tf.placeholder(tf.float32, shape=[None, z_dim], name="z")
  g_sample = _generator(z)
  d_real, dlogits_real = _discriminator(X)
  d_fake, dlogits_fake = _discriminator(g_sample, reuse=True)

  # Alternative loss formulation
  #
  # d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
  # g_loss = -tf.reduce_mean(tf.log(d_fake))

  d_loss_real = sigmoid_loss(dlogits_real, tf.ones_like(dlogits_real))
  d_loss_fake = sigmoid_loss(dlogits_fake, tf.zeros_like(dlogits_fake))
  d_loss = d_loss_real + d_loss_fake
  g_loss = sigmoid_loss(dlogits_fake, tf.ones_like(dlogits_fake))
  t_vars = tf.trainable_variables()
  d_vars = [v for v in t_vars if "discriminator" in v.name]
  g_vars = [v for v in t_vars if "generator" in v.name]
  d_opt = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
  g_opt = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)

  return dict(
      X=X,
      z=z,
      g_sample=g_sample,
      g_loss=g_loss,
      d_loss=d_loss,
      d_opt=d_opt,
      g_opt=g_opt)


def init(sess):
  sess.run(tf.global_variables_initializer())


def optimize_discriminator(sess, var_dict, X, z):
  _, d_loss = sess.run(
      [var_dict["d_opt"], var_dict["d_loss"]],
      feed_dict={var_dict["X"]: X,
                 var_dict["z"]: z})
  return d_loss


def optimize_generator(sess, var_dict, z):
  _, d_loss = sess.run(
      [var_dict["g_opt"], var_dict["g_loss"]], feed_dict={var_dict["z"]: z})
  return d_loss


def sample_from_generator(sess, var_dict, z):
  return sess.run(
      var_dict["g_sample"],
      feed_dict={var_dict["z"]: z})

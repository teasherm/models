import tensorflow as tf

from lib.ops import xavier_initializer, sigmoid_loss

relu = tf.nn.relu


def _sample_z(mu, log_var):
  eps = tf.random_normal(shape=tf.shape(mu))
  return mu + tf.exp(log_var / 2) * eps


def _Q(X, z_dim, h_dim=128):
  with tf.variable_scope("Q"):
    h = tf.layers.dense(
        X,
        h_dim,
        activation=relu,
        kernel_initializer=xavier_initializer(X),
        name="layer1")
    z_mu = tf.layers.dense(
        h,
        z_dim,
        activation=relu,
        kernel_initializer=xavier_initializer(h),
        name="mu")
    z_logvar = tf.layers.dense(
        h,
        z_dim,
        activation=relu,
        kernel_initializer=xavier_initializer(h),
        name="sigma")
    return z_mu, z_logvar


def _P(z, X_dim, h_dim=128, reuse=False):
  with tf.variable_scope("P") as scope:
    if reuse: scope.reuse_variables()
    h = tf.layers.dense(
        z, h_dim,
        activation=relu,
        kernel_initializer=xavier_initializer(z),
        name="layer1")
    logits = tf.layers.dense(
        h,
        X_dim,
        activation=relu,
        kernel_initializer=xavier_initializer(h),
        name="logits")
    prob = tf.nn.sigmoid(logits)
    return prob, logits


def build_graph(X_dim=784, z_dim=100):
  X = tf.placeholder(tf.float32, shape=[None, X_dim])
  z = tf.placeholder(tf.float32, shape=[None, z_dim])

  z_mu, z_logvar = _Q(X, z_dim)
  z_sample = _sample_z(z_mu, z_logvar)
  _, logits = _P(z_sample, X_dim)

  X_samples, _ = _P(z, X_dim, reuse=True)

  # E[log P(X|z)]
  recon_loss = sigmoid_loss(logits, X, axis=1)

  # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
  kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)

  vae_loss = tf.reduce_mean(recon_loss + kl_loss)
  train_op = tf.train.AdamOptimizer().minimize(vae_loss)

  return dict(
      X=X, z=z, X_samples=X_samples, vae_loss=vae_loss, train_op=train_op)


def init(sess):
  sess.run(tf.global_variables_initializer())


def train(sess, var_dict, X):
  _, loss = sess.run([var_dict["train_op"], var_dict["vae_loss"]],
                     {var_dict["X"]: X})
  return loss


def sample(sess, var_dict, z):
  return sess.run(
      var_dict["X_samples"],
      feed_dict={var_dict["z"]: z})

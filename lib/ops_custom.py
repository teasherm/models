import tensorflow as tf

from lib.ops import activation_summaries, variable_summaries


def dense_custom(input_tensor,
                 input_dim,
                 output_dim,
                 layer_name='dense',
                 act=tf.nn.relu,
                 summarize=False):
  with tf.variable_scope(layer_name):
    with tf.variable_scope('weights'):
      W = tf.get_variable(
          'weights',
          shape=[input_dim, output_dim],
          initializer=tf.truncated_normal_initializer(stddev=0.1))
      if summarize: variable_summaries(W)
    with tf.variable_scope('biases'):
      b = tf.get_variable(
          'bias', shape=[output_dim], initializer=tf.constant_initializer(0.1))
      if summarize: variable_summaries(b)
    with tf.name_scope('preactivations'):
      preactivations = tf.matmul(input_tensor, W) + b
      if summarize: tf.summary.histogram('preactivations', preactivations)
    with tf.name_scope('activations'):
      activations = act(preactivations)
      if summarize: activation_summaries(activations)

  return activations


def conv2d_custom(layer_name,
                  input_,
                  in_channels=None,
                  out_channels=None,
                  filter_size=None,
                  stride=None,
                  act=tf.nn.relu,
                  summarize=False):
  with tf.variable_scope(layer_name) as scope:
    with tf.variable_scope('weights'):
      kernel = tf.get_variable(
          'weights', [filter_size, filter_size, in_channels, out_channels],
          initializer=tf.truncated_normal_initializer(stddev=5e-2))
      if summarize: variable_summaries(kernel)
    with tf.variable_scope('biases'):
      biases = tf.get_variable(
          'biases', [out_channels], initializer=tf.constant_initializer(0.1))
      if summarize: variable_summaries(biases)
    with tf.name_scope('preactivations'):
      conv = tf.nn.conv2d(
          input_, kernel, strides=[1, stride, stride, 1], padding='SAME')
      preactivations = tf.nn.bias_add(conv, biases)
      if summarize: variable_summaries(preactivations)
    with tf.name_scope('activations'):
      activations = act(preactivations, name=scope.name)
      if summarize: activation_summaries(activations)
    return activations

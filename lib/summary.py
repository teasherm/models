import tensorflow as tf


def activation_summaries(var):
  with tf.name_scope('activation_summaries'):
    tf.summary.histogram('activations', var)
    tf.summary.scalar('sparsity', tf.nn.zero_fraction(var))


def variable_summaries(var):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

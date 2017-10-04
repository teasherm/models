import tensorflow as tf


def sigmoid_loss(logits, labels, axis=None):
  return tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels),
      axis=axis)


def xavier_initializer(in_tensor):
  in_dim = in_tensor.get_shape().as_list()[1]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal_initializer(stddev=xavier_stddev)


def batch_normalization(x,
                        name="bnorm",
                        epsilon=1e-5,
                        momentum=0.9,
                        training=True):
  return tf.layers.batch_normalization(
      x, momentum=momentum, epsilon=epsilon, training=training, name=name)


def leaky_relu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak * x, name=name)


def embedding_layer(input_tensor, embeddings, reshaped=[-1]):
  with tf.variable_scope('emb'):
    W = tf.get_variable(
        'weights',
        shape=embeddings.shape,
        initializer=tf.constant_initializer(embeddings),
        trainable=False)
    embedded_input = tf.nn.embedding_lookup(W, input_tensor)
    embedded_input_reshaped = tf.reshape(embedded_input, reshaped)

  return embedded_input_reshaped


def conv2d(inputs,
           filters,
           kernel_size=(5, 5),
           strides=(2, 2),
           padding="same",
           activation=tf.nn.relu,
           kernel_initializer=tf.contrib.layers.xavier_initializer(),
           name=None):
  return tf.layers.conv2d(
      inputs,
      filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      activation=activation,
      kernel_initializer=kernel_initializer,
      name=name)


def deconv2d(inputs,
             filters,
             kernel_size=(5, 5),
             strides=(2, 2),
             padding="same",
             activation=tf.nn.relu,
             kernel_initializer=tf.contrib.layers.xavier_initializer(),
             name=None):
  return tf.layers.conv2d_transpose(
      inputs,
      filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      activation=activation,
      kernel_initializer=kernel_initializer,
      name=name)


def dense(inputs,
          units,
          kernel_initializer=tf.contrib.layers.xavier_initializer(),
          activation=tf.nn.relu,
          name=None):
  return tf.layers.dense(
      inputs,
      units,
      activation=activation,
      kernel_initializer=kernel_initializer,
      name=name)

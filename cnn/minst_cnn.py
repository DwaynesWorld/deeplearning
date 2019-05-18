import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


def init_weights(shape):
    init_rand_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_rand_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W):
    """
    Wrapper for TF conv2d\n
    x --> [batch, in_height, in_width, in_channels]\n
    W --> [filter_height, filter_width, in_channels, out_channels]
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    """
    Wrapper for TF max_pool\n
    x --> [batch, in_height, in_width, in_channels]
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W,) + b)


def full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return (tf.matmul(input_layer, W) + b)


def conv_net(x):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = conv_layer(x_image, [5, 5, 1, 32])
    pool1 = maxpool2d(conv1)
    conv2 = conv_layer(pool1, [5, 5, 32, 64])
    pool2 = maxpool2d(conv2)
    flattened = tf.reshape(pool2, [-1, 7*7*64])
    full1 = tf.nn.relu(full_layer(flattened, 1024))
    full1_dropout = tf.nn.dropout(full1, keep_prob)
    return full_layer(full1_dropout, 10)


x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

y_pred = conv_net(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=y_true, logits=y_pred
))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
batch_size = 50
steps = 5000

with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed = {
            x: batch_x,
            y_true: batch_y,
            keep_prob: 0.5
        }
        sess.run(train, feed_dict=feed)

        if i % 100 == 0:
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))
            feed = {
                x: mnist.test.images,
                y_true: mnist.test.labels,
                keep_prob: 1.0
            }
            result = sess.run(acc, feed_dict=feed)

            print("Step: {}".format(i))
            print("Accuracy: {}\n".format(result))

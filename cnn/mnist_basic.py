import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Read mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Examine mnist data
# print(type(mnist))
# print(mnist.train.images)
# print(mnist.train.num_examples)
# print(mnist.test.num_examples)

# Examine images
# print(mnist.train.images.shape)
# single_image = mnist.train.images[1].reshape(28, 28)
# print(single_image)
# print(single_image.min())
# print(single_image.max())
# plt.imshow(single_image, cmap='gist_gray')
# plt.show()

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])

# Variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Create graph operations
y = tf.matmul(x, W) + b

# Loss Function
y_true = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# Create Session and Run
init = tf.global_variables_initializer()
batch_size = 100
epochs = 1000

with tf.Session() as sess:
    # Start training session
    sess.run(init)

    for step in range(epochs):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

    # Get accuracy
    prediction = tf.argmax(y, axis=1)
    actual = tf.argmax(y_true, axis=1)
    correct_prediction = tf.equal(prediction, actual)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(acc, feed_dict={x: mnist.test.images, y_true: mnist.test.labels})
    print(result)

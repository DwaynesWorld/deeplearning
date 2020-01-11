import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# from prettyprinter import cpprint
from utils import read_cifar_datasets, preview_transformed_data, Cifar

# Import CIFAR data
cifar_data = read_cifar_datasets('./CIFAR_data/')

# Preview transformed data
# preview_transformed_data(cifar_data['data_batch1'][b'data'], True)

# Import and setup Cifar helper class
cifar = Cifar(cifar_data)
cifar.set_up_images()

# Training Parameters
learning_rate = 0.001
num_steps = 5000
batch_size = 100
display_step = 50

# Network Parameters
input_shape = [None, 32, 32, 3]
num_classes = 10  # CIFAR total classes
# dropout = 0.5  # Dropout, probability to keep units

# Network Inputs
X = tf.placeholder(tf.float32, shape=input_shape)
Y = tf.placeholder(tf.float32, shape=[None, num_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Store layers weight & bias
weights = {
    # 4x4 conv, 3 inputs, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([4, 4, 3, 32], stddev=0.1)),

    # 4x4 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.1)),

    # fully connected, 8*8*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.truncated_normal([8*8*64, 1024], stddev=0.1)),

    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([1024, num_classes], stddev=0.1))
}

biases = {
    'bc1': tf.Variable(tf.constant(0.1, shape=[32])),
    'bc2': tf.Variable(tf.constant(0.1, shape=[64])),
    'bd1': tf.Variable(tf.constant(0.1, shape=[1024])),
    'out': tf.Variable(tf.constant(0.1, shape=[num_classes]))
}


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    """Conv2D wrapper, with bias and relu activation"""
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    # x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x + b)


def maxpool2d(x, k=2):
    """MaxPool2D wrapper"""
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def conv_net(x, weights, bias, dropout):
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    pool1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(pool2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = cifar.next_batch(batch_size)

        # Run optimization op (backprop)
        train_feed = {X: batch_x, Y: batch_y, keep_prob: 0.5}
        sess.run(train_op, feed_dict=train_feed)

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            eval_feed = {X: batch_x, Y: batch_y, keep_prob: 1.0}
            loss, acc = sess.run([loss_op, accuracy], feed_dict=eval_feed)

            print("Step " + str(step) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for test images
    test_feed = {X: cifar.test_images, Y: cifar.test_labels, keep_prob: 1.0}
    print("Testing Accuracy:", sess.run(accuracy, feed_dict=test_feed))

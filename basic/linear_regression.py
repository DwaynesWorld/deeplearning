import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Init Seeds
np.random.seed(101)
tf.set_random_seed(101)

# # Example Linear Regression
# n_features = 10
# n_dense_neurons = 3

# # data
# x = tf.placeholder(tf.float64, (None, n_features))

# # Weights: initially random
# weights = np.random.random((n_features, n_dense_neurons))
# print("weights: ", weights)
# W = tf.Variable(weights)

# # Bias: initially 1
# ones = np.ones(n_dense_neurons)
# print("ones: ", ones)
# b = tf.Variable(ones)

# # Regression Formula: y = mx + b
# xW = tf.matmul(x, W)
# z = tf.add(xW, b)

# # Activation Func: Sigmoid
# a = tf.sigmoid(z)

# init = tf.global_variables_initializer()

# data = np.random.random([1, n_features])
# print("data: ", data)

# with tf.Session() as sess:
#     sess.run(init)
#     layer_out = sess.run(a, feed_dict={x: data})
#     print(layer_out)

x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
print(x_data)

y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
print(y_label)

# plt.plot(x_data, y_label, "*")
# plt.show()

# y = mx+b
# print(np.random.rand(2))
m = tf.Variable(0.68530633)
b = tf.Variable(0.51786747)

error = 0

for x, y in zip(x_data, y_label):
    y_hat = m*x + b
    error += (y-y_hat)**2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    training_steps = 100
    for i in range(training_steps):
        sess.run(train)

    final_slope, final_intercept = sess.run([m, b])

x_test = np.linspace(-1, 11, 10)
y_pred_plot = final_slope*x_test + final_intercept

plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, "*")
plt.show()

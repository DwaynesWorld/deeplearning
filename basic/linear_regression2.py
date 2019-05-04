from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

x_data = np.linspace(0.0, 10.0, 1000000)
# print(x_data)
noise = np.random.randn(len(x_data))
# print(noise)

y_true = (0.5 * x_data) + 5 + noise
# print(y_true)

x_df = pd.DataFrame(data=x_data, columns=['X Data'])
# print(x_df.head())

y_df = pd.DataFrame(data=y_true, columns=['Y'])
# print(y_df.head())

my_data = pd.concat([x_df, y_df], axis=1)
# print(my_data)

sample = my_data.sample(n=250)
# sample.plot(kind='scatter', x='X Data', y='Y')
# print(sample)
# plt.show()

############### Manual Linear Regression ###############
# batch_size = 8

# m = tf.Variable(np.random.randn())
# b = tf.Variable(np.random.randn())

# x = tf.placeholder(tf.float32, [batch_size])
# y = tf.placeholder(tf.float32, [batch_size])

# y_model = m*x + b
# error = tf.reduce_sum(tf.square(y - y_model))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# train = optimizer.minimize(error)

# init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)
#     batches = 10000
#     # training
#     for i in range(batches):
#         rand_idx = np.random.randint(len(x_data), size=batch_size)
#         feed = {x: x_data[rand_idx], y: y_true[rand_idx]}
#         sess.run(train, feed_dict=feed)
#     # predictive model
#     model_m, model_b = sess.run([m, b])

# print(model_m)
# print(model_b)

# y_hat = x_data*model_m + model_b
# my_data.sample(250).plot(kind='scatter', x='X Data', y='Y')
# plt.plot(x_data, y_hat, 'r')
# plt.show()

############### TensorFlow Linear Regression ###############

feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]

estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

x_train, x_eval, y_train, y_eval = train_test_split(
    x_data,
    y_true,
    test_size=0.3,
    random_state=101)

# print(x_train.shape)
# print(x_eval.shape)

input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train},
    y_train,
    batch_size=8,
    num_epochs=None,
    shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train},
    y_train,
    batch_size=8,
    num_epochs=1000,
    shuffle=False)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {'x': x_eval},
    y_eval,
    batch_size=8,
    num_epochs=1000,
    shuffle=False)

# Train random sample of training data
estimator.train(input_fn=input_fn, steps=1000)

# Get eval metrics based on all training data
train_metrics = estimator.evaluate(input_fn=train_input_fn, steps=1000)

# get eval metrics on test data
eval_metrics = estimator.evaluate(input_fn=eval_input_fn, steps=1000)

# Overfitting can happen when train data loss is not close to eval data loss
print("Training Data Metrics")
print(train_metrics)

print("Eval Data Metrics")
print(eval_metrics)

# Predictive Model
new_data = np.linspace(0, 10, 10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn(
    {'x': new_data},
    shuffle=False)

# print(list(estimator.predict(input_fn=input_fn_predict)))

predictions = []
for pred in estimator.predict(input_fn=input_fn_predict):
    predictions.append(pred['predictions'])

# print(predictions)
my_data.sample(250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(new_data, predictions, 'r')
plt.show()

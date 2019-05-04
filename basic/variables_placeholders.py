import tensorflow as tf

sess = tf.InteractiveSession()

tensor = tf.random_uniform((4, 4), 0, 1)
print(tensor)

var = tf.Variable(tensor)
print(var)

init = tf.global_variables_initializer()
sess.run(init)

result = sess.run(var)
print(result)

ph = tf.placeholder(tf.float32, shape=(None, 6))

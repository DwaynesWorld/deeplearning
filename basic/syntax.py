import tensorflow as tf

print(tf.__version__)

hello = tf.constant("Hello ")
world = tf.constant("World")

with tf.Session() as sess:
    result = sess.run(hello + world)

print(result)

a = tf.constant(10)
b = tf.constant(20)
c = a + b
print(type(c))

with tf.Session() as sess:
    result = sess.run(c)

print(result)

const = tf.constant(10)
fill_mat = tf.fill((4, 4), 10)
zeroes = tf.zeros((4, 4))
ones = tf.ones((4, 4))
randn = tf.random_normal((4, 4), mean=0, stddev=1.0)
randu = tf.random_uniform((4, 4), minval=0, maxval=1)

ops = [const, fill_mat, zeroes, ones, randn, randu]

sess = tf.InteractiveSession()

print("Operations")
for op in ops:
    print(sess.run(op))
    print('\n')

a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[10], [100]])

print(a.get_shape())
print(b.get_shape())

result = tf.matmul(a, b)
print(result.eval())

result = sess.run(result)
print(result)

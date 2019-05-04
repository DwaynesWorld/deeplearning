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

# rand_a = np.random.uniform(0, 100, (5, 5))
# print(rand_a)

# rand_b = np.random.uniform(0, 100, (5, 1))
# print(rand_b)


# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)

# add_op = a + b
# mul_op = a * b

# with tf.Session() as sess:
#     result = sess.run(add_op, feed_dict={a: rand_a, b: rand_b})
#     print(result)
#     print("\n")

#     result = sess.run(mul_op, feed_dict={a: rand_a, b: rand_b})
#     print(result)

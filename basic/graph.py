import tensorflow as tf

n1 = tf.constant(1)
n2 = tf.constant(2)
n3 = n1 + n2
with tf.Session() as sess:
    result = sess.run(n3)

print(result)
print(tf.get_default_graph())

g1 = tf.get_default_graph()
print(g1)

g2 = tf.Graph()
print(g2)

with g2.as_default():
    print(g2 is tf.get_default_graph())

print(g2 is tf.get_default_graph())

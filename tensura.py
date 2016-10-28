import tensorflow as tf
import numpy as np
# x1 = tf.constant(5)
# x2 = tf.constant(6)
# x3 = tf.constant([[3.,3.]])
# x4 = tf.constant([[2.],[2.]])
#result = tf.mul(x1, x2)
#result = tf.matmul(x3, x4)
#print(result)
# with tf.Session() as sess:
#      print(sess.run(result))

X = np.array([[1, 1], [3, 0], [4, 1], [2, 0]])
v = np.zeros(X.shape[1])
y = [1, 2, 3, 4]
#y = np.zeros(9)
for i, j in zip(X, y):
    print("{} <===> {}".format(i, j))
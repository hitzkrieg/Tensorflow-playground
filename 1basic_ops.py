import tensorflow as tf
import numpy as np

a = tf.placeholder("float") 
b = tf.placeholder("float") 
c = tf.mul(a, b) 
data = np.random.randint(10, size=5)
print(data)
x = tf.constant(data, name='x')
y = tf.Variable(5*x**2 -3*x + 15, name='y')

model = tf.global_variables_initializer()

with tf.Session() as sess: 
	sess.run(model)
	print("%f should equal 2.0" % sess.run(c, feed_dict={a: 1, b: 2})) 
	print(sess.run(y))










import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#y= 2x + 3 say is the dataset we try to create with some random error

n= 101


xs = np.linspace(-3, 3, n)

ys = 2*xs + 3 + np.random.uniform(-0.5, 0.5, n)



X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
Y_pred = tf.add(tf.mul(X, W), b)

cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n)

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

epoch= 100
with tf.Session() as sess:
    
	sess.run(tf.initialize_all_variables())

    
	for epoch_i in range(epoch):
		sess.run(optimizer, feed_dict={X: xs, Y: ys})

		training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})
		print(training_cost)

     
	plt.scatter(xs,ys,color='black')
	plt.plot(xs, Y_pred.eval(feed_dict={X: xs}, session=sess), color='blue', linewidth=3)
	plt.xticks(())
	plt.yticks(())
	plt.show()

	print(sess.run(W), sess.run(b))    

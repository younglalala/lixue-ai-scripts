import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 28,28,1])
y_ = tf.placeholder(tf.float32, [None,10])
dp = tf.placeholder(tf.float32)

conv1_w = tf.Variable(tf.random_normal([3,3,1,32],dtype=tf.float32,stddev=0.1))
conv1_b1 = tf.Variable(tf.zeros([32]))
conv1 = tf.nn.relu(tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='SAME')+conv1_b1)

pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') #14*14

conv2_w = tf.Variable(tf.random_normal([3,3,32,64],dtype=tf.float32,stddev=0.1))
conv2_b2 = tf.Variable(tf.zeros([64]))
conv2 = tf.nn.relu(tf.nn.conv2d(pool1, conv2_w, strides=[1, 1, 1, 1], padding='SAME')+conv2_b2)

pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') #7*7

flat = tf.reshape(pool2,[-1,7*7*64])

W1 = tf.Variable(tf.random_normal([7*7*64,1024],stddev=0.1,dtype=tf.float32))
b1 = tf.Variable(tf.zeros([1024]))
W2 = tf.Variable(tf.random_normal([1024,10],stddev=0.1,dtype=tf.float32))
b2 = tf.Variable(tf.zeros([10]))

y1 = tf.nn.relu(tf.matmul(flat,W1) + b1)
y1 = tf.nn.dropout(y1,keep_prob=dp)
y = tf.nn.softmax(tf.matmul(y1 ,W2) + b2)

cross_entropy =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))


optimizer = tf.train.AdamOptimizer(0.001)
train_step = optimizer.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
rst = tf.cast(correct_prediction, "float")
accuracy = tf.reduce_mean(rst)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # saver.restore(sess,"save/xx.dpk")


    for i in range(10000):
        batch_xs_x, batch_ys = mnist.train.next_batch(50)
        batch_xs = batch_xs_x.reshape([50, 28, 28, 1])

        ac,_,loss = sess.run([accuracy,train_step,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys,dp:0.5})
        print('第{}次的误差是{}，精度是{}'.format(i,loss,ac))
        # if i%100 == 0:
        #     saver.save(sess,"save/xx.dpk")

        if i % 100 == 0:
            batch_xs_x1 ,batch_ys1 = mnist.test.next_batch(50)
            batch_xs1 = batch_xs_x1.reshape([-1, 28, 28, 1])
            test_ac,test_loss=sess.run([accuracy,cross_entropy], feed_dict={x:  batch_xs1, y_: batch_ys1,dp:1.})
            print('测试集第{}次的误差是:{}，精度是:{}'.format(i,test_loss,test_ac))
            # print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


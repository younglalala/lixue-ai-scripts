import tensorflow as tf
from num_modle.onehott import *
from mobile_num_modle1.train_sample import *
from mobile_num_modle1.test_sample import *
import  numpy as np
import scipy.misc


# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.misc
# from num_modle.onehott import *
# from num_modle.recognition_sample import *
# from num_modle.test_sample import *
# import tempfile
# import subprocess
# tf.contrib.lite.tempfile = tempfile
# tf.contrib.lite.subprocess = subprocess

class RecingnitionNum:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='input')
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        self.conv1_w = tf.Variable(tf.truncated_normal([5, 5, 1, 16], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.conv1_b = tf.Variable(tf.zeros([16]))
        self.conv1_dw = tf.Variable(tf.truncated_normal([3, 3, 16, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.conv1_db = tf.Variable(tf.zeros([16]))

        self.conv2_w = tf.Variable(tf.random_normal([5, 5, 16, 32], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv2_b = tf.Variable(tf.zeros([32]))
        self.conv2_dw = tf.Variable(tf.truncated_normal([1, 1, 32, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv2_db = tf.Variable(tf.zeros([32]))

        # self.conv3_w = tf.Variable(tf.random_normal([3, 3, 32, 64], dtype=tf.float32, stddev=tf.sqrt(1 / 64)))
        # self.conv3_b = tf.Variable(tf.zeros([64]))
        # self.conv3_dw = tf.Variable(tf.truncated_normal([1, 1, 64, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 64)))
        # self.conv3_db = tf.Variable(tf.zeros([64]))

        self.fc1_w = tf.Variable(tf.truncated_normal([2 * 2 * 32, 256], dtype=tf.float32, stddev=tf.sqrt(1 / 256)))
        self.fc1_b = tf.Variable(tf.zeros([256]))

        self.fc3_w = tf.Variable(tf.truncated_normal([256, 10], dtype=tf.float32, stddev=tf.sqrt(1 / 10)))
        self.fc3_b = tf.Variable(tf.zeros([10]))

    def forward(self):
        self.conv1 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(self.x, self.conv1_w, strides=[1, 2, 2, 1],
                                                                           padding='SAME') + self.conv1_b))  # hape=(?, 30, 27, 10)

        self.conv2w = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.depthwise_conv2d(self.conv1, self.conv1_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv1_db))

        self.conv2 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv2w, self.conv2_w, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_b))
        self.conv2w = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.depthwise_conv2d(self.conv2, self.conv2_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_db))
        # self.conv3 = tf.nn.relu(tf.layers.batch_normalization(
        #     tf.nn.conv2d(self.conv2w, self.conv3_w, strides=[1, 1, 1, 1],padding='SAME') + self.conv3_b))  # (?, 7, 7, 64)
        # self.conv3w=tf.nn.relu(tf.layers.batch_normalization(
        #         tf.nn.depthwise_conv2d(self.conv3, self.conv3_dw, strides=[1, 2, 2, 1],padding='SAME') + self.conv3_db))

        self.pool = tf.nn.max_pool(self.conv2w, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(self.pool)

        flat = tf.reshape(self.pool, [-1, 2 * 2 * 32])

        self.fc1 = tf.nn.relu(tf.matmul(flat, self.fc1_w) + self.fc1_b)



        self.out = tf.matmul(self.fc1, self.fc3_w) + self.fc3_b
        self.output=tf.reshape(self.out,[-1,10],name='output')

    def backward(self):


        self.cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.output))
        # global_step = tf.Variable(0, trainable=False)
        # self.learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.96, staircase=True)


        self.optimizer=tf.train.AdamOptimizer(0.0004).minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y_, 1))
        self.out_argmax=tf.argmax(self.output, 1)
        self.out_argmax1=tf.reshape(self.out_argmax,[-1])
        self.lable_=tf.argmax(self.y_, 1)
        self.rst = tf.cast(self.correct_prediction, "float")
        self.accuracy = tf.reduce_mean(self.rst)


if __name__=='__main__':
    net=RecingnitionNum()
    net.forward()
    net.backward()

    init = tf.global_variables_initializer()

    train = train_shuffle_batch(train_filename, [64, 64, 1], 128)
    test = test_shuffle_batch(test_filename, [64, 64, 1],1000)
    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()      # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        sess.run(init)
        x=[]
        y=[]

        saver.restore(sess,"num_save/train.dpk")
        yy=0
        flag = 0
        flag1 = 0
        for ii in range(100000):
            train_img1, train_label1 = sess.run(train)
            train_img11=train_img1/255-0.5
            train_label1=one_hot(9,train_label1.reshape([-1]).tolist())

            train_out,train_la,train_loss,train_ac,_=sess.run([net.out_argmax1,net.lable_,net.cross_entropy,net.accuracy,net.optimizer],
                                                              feed_dict={net.x:train_img11,net.y_:train_label1})

            train_f='/Users/wywy/Desktop/num_train_flase'
            train_c=0
            for oo in range(len(train_out)):
                if train_out.tolist()[oo] ==train_la.tolist()[oo]:
                    pass
                # else:
                #     # scipy.misc.imsave(train_f+'/'+str(train_file1[oo][0])+'_'+str(train_out.tolist()[oo])+'_'+str(train_la.tolist()[oo])+'.jpg',
                #     #                   train_img1.reshape([-1,64,64])[oo])

                    # print(train_f+'/'+str(train_c)+'_'+str(train_out.tolist()[oo])+'_'+str(train_la.tolist()[oo])+'.jpg')
                    train_c+=1



            print('第{}次误差是：{}，精度是：{}，out:{},label:{}'.format(ii,train_loss,train_ac,train_out[0],train_la[0]))
            x.append(ii)
            y.append(train_loss)
            plt.plot(x,y,'red')
            plt.pause(0.01)
            plt.clf()
            if ii%50==0:
                # test_img1, test_label1, test_file1 = sess.run(test)
                # test_img11 = test_img1 / 255 - 0.5
                # test_label1 = one_hot(9, test_label1.reshape([-1]).tolist())
                #
                # test_out, test_la, test_loss, test_ac = sess.run(
                #     [net.out_argmax1, net.lable_, net.cross_entropy, net.accuracy],
                #     feed_dict={net.x: test_img11, net.y_: test_label1})
                # print('第{}次测试集误差是:{}，精度为:{}'.format(ii,test_loss,test_ac))
                # test_f = '/Users/wywy/Desktop/num_test_false'
                # test_c = 0
                # for oo1 in range(len(test_out)):
                #     if test_out.tolist()[oo1] == test_la.tolist()[oo1]:
                #         pass
                #     else:
                #         scipy.misc.imsave(test_f + '/' + str(test_file1[oo1][0]) + '_' + str(test_out.tolist()[oo1]) + '_' + str(
                #             test_la.tolist()[oo1]) + '.jpg',
                #                           test_img1.reshape([-1,64,64])[oo1])
                #
                #         # print(test_f + '/' + str(test_c) + '_' + str(test_out.tolist()[oo1]) + '_' + str(
                #         #     test_la.tolist()[oo1]) + '.jpg')
                #         test_c+=1
                saver.save(sess, "num_save/train.dpk")






























# class MobileModle:
#     def __init__(self):
#         self.x=tf.placeholder(shape=[None,64,64,3],dtype=tf.float32)
#         self.y_=tf.placeholder(shape=[None,10],dtype=tf.float32)
#
#         self.fc_w=tf.Variable(tf.random_normal([4*4*64,256],dtype=tf.float32))
#         self.fc_b=tf.Variable(tf.zeros([256]))
#
#         self.fc2_w=tf.Variable(tf.random_normal([256,10],dtype=tf.float32))
#         self.fc2_b=tf.Variable(tf.zeros([10]))
#
#     def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
#         with tf.variable_scope(name):
#             kernel = tf.get_variable(name='W',
#                                      shape=[filter_size, filter_size, in_channels, out_channels],
#                                      dtype=tf.float32,
#                                      initializer=tf.random_normal_initializer())  # tf.glorot_normal_initializer
#
#             b = tf.get_variable(name='b',
#                                 shape=[out_channels],
#                                 dtype=tf.float32,
#                                 initializer=tf.zeros_initializer())
#
#             con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME') + b
#
#         return con2d_op
#     def forward(self):
#         self.conv1=tf.nn.relu(tf.layers.batch_normalization (self._conv2d(self.x,'conv1',3,3,10,1)))
#         self.pool=tf.nn.max_pool(self.conv1,[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#
#         self.conv2=tf.nn.relu(tf.layers.batch_normalization(self._conv2d(self.pool,'conv2',3,10,16,1)))
#         self.pool2=tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#
#         self.conv3=tf.nn.relu(tf.layers.batch_normalization(self._conv2d(self.pool2,'conv3',3,16,64,1)))
#         self.pool3=tf.nn.max_pool(self.conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#
#         self.conv4 = tf.nn.relu(tf.layers.batch_normalization(self._conv2d(self.pool3, 'conv4', 3, 64, 64, 1)))
#         self.pool4 = tf.nn.max_pool(self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
#
#         self.flat=tf.reshape(self.pool4,[-1,4*4*64])
#         self.fc1=tf.nn.relu( tf.matmul( self.flat,self.fc_w)+self.fc_b)
#         self.out=tf.matmul(self.fc1,self.fc2_w)+self.fc2_b
#
#     def backward(self):
#         self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out,labels=self.y_))
#
#         self.op=tf.train.AdamOptimizer(0.001).minimize(self.loss)
#         self.correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y_, 1))
#         self.out_argmax = tf.argmax(self.out, 1)
#         self.out_argmax1 = tf.reshape(self.out_argmax, [-1])
#         self.lable_ = tf.argmax(self.y_, 1)
#         self.rst = tf.cast(self.correct_prediction, "float")
#         self.accuracy = tf.reduce_mean(self.rst)
#
#
#
# if __name__=='__main__':
#     a=one_hot(3,[1,2,3])
#     net=MobileModle()
#     net.forward()
#     net.backward()
#     init = tf.global_variables_initializer()
#
#     train_data = train_shuffle_batch(train_filename, [64, 64, 3], 100)
#     test_data = test_shuffle_batch(test_filename, [64, 64, 3], 1000)
#     saver = tf.train.Saver(max_to_keep=5)
#     with tf.Session() as sess:
#         coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
#         threads = tf.train.start_queue_runners(coord=coord, sess=sess)
#         sess.run(init)
#         x = []
#         y = []
#
#         for ii in range(10000):
#             train_img1, train_label1, train_file1 = sess.run(train_data)
#             train_img11=train_img1/255-0.5
#             train_label1=one_hot(9,train_label1.reshape([-1]).tolist())
#
#             train_out,train_la,train_loss,train_ac,_=sess.run([net.out_argmax1,net.lable_,net.loss,net.accuracy,net.op],
#                                                               feed_dict={net.x:train_img11,net.y_:train_label1})
#
#
#             print('误差是：{}，精度是：{}，out:{},label:{}'.format(train_loss,train_ac,train_out[0],train_la[0]))
#             if ii%1000==0:
#                 saver.save(sess, "num_save/train.dpk")







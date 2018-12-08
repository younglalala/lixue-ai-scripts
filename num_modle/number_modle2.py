import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from num_modle.onehott import *
from num_modle.recognition_sample import *
from num_modle.test_sample import *
import tempfile
import subprocess
tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess

class RecingnitionNum:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[1, 64, 64, 3], name='input')
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[1, 11])

        self.conv1_w = tf.Variable(tf.truncated_normal([3, 3, 3, 32], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.conv1_b = tf.Variable(tf.zeros([32]))
        self.conv1_dw = tf.Variable(tf.truncated_normal([3, 3, 32, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.conv1_db = tf.Variable(tf.zeros([32]))

        self.conv2_w = tf.Variable(tf.random_normal([3, 3, 32, 64], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv2_b = tf.Variable(tf.zeros([64]))
        self.conv2_dw = tf.Variable(tf.truncated_normal([3, 3, 64, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv2_db = tf.Variable(tf.zeros([64]))

        self.conv3_w = tf.Variable(tf.random_normal([1, 1, 64, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.conv3_b = tf.Variable(tf.zeros([128]))

        self.conv4_w = tf.Variable(tf.random_normal([1, 1, 128, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.conv4_b = tf.Variable(tf.zeros([128]))

        self.conv5_w = tf.Variable(tf.random_normal([1, 1, 128, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 128)))
        self.conv5_b = tf.Variable(tf.zeros([128]))

        self.fc1_w = tf.Variable(tf.truncated_normal([2 * 2 * 128, 256], dtype=tf.float32, stddev=tf.sqrt(1 / 256)))
        self.fc1_b = tf.Variable(tf.zeros([256]))

        self.fc3_w = tf.Variable(tf.truncated_normal([256, 11], dtype=tf.float32, stddev=tf.sqrt(1 / 11)))
        self.fc3_b = tf.Variable(tf.zeros([11]))

    def forward(self):
        self.conv1 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(self.x, self.conv1_w, strides=[1, 2, 2, 1],
                                                                           padding='SAME') + self.conv1_b))  # hape=(?, 30, 27, 10)

        self.conv2w = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.depthwise_conv2d(self.conv1, self.conv1_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv1_db))

        self.conv2 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.conv2w, self.conv2_w, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_b))
        self.conv2w = tf.nn.relu(
            tf.nn.depthwise_conv2d(self.conv2, self.conv2_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_db)
        self.conv3 = tf.nn.relu(tf.nn.conv2d(self.conv2w, self.conv3_w, strides=[1, 1, 1, 1],
                                             padding='SAME') + self.conv3_b)  # (?, 7, 7, 64)
        self.conv4=tf.nn.relu(tf.nn.conv2d(self.conv3,self.conv4_w,strides=[1,1,1,1],padding='SAME')+self.conv4_b)
        self.conv5=tf.nn.relu(tf.nn.conv2d(self.conv4,self.conv5_w,strides=[1,1,1,1],padding='SAME')+self.conv5_b)

        self.pool = tf.nn.max_pool(self.conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print(self.pool)

        flat = tf.reshape(self.pool, [-1, 2 * 2 * 128])

        self.fc1 = tf.nn.relu(tf.matmul(flat, self.fc1_w) + self.fc1_b)



        self.out = tf.matmul(self.fc1, self.fc3_w) + self.fc3_b
        self.output=tf.reshape(self.out,[-1,11],name='output')

    def backward(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.output))
                                   #        + 0.0001 * tf.reduce_sum(tf.pow(self.conv1_w, 2))+ 0.0001 * tf.reduce_sum(tf.pow(self.conv1_dw, 2))
                                   # + 0.0001 * tf.reduce_sum(tf.pow(self.conv2_w, 2))+ 0.0001 * tf.reduce_sum(tf.pow(self.conv2_dw, 2))
                                   # + 0.0001 * tf.reduce_sum(tf.pow(self.fc1_w, 2))
                                   # + 0.0001 * tf.reduce_sum(tf.pow(self.fc3_w, 2)))
        # global_step = tf.Variable(0, trainable=False)
        # self.learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.96, staircase=True)


        self.optimizer=tf.train.AdamOptimizer(0.0002).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y_, 1))
        self.out_argmax=tf.argmax(self.output, 1)
        self.out_argmax1=tf.reshape(self.out_argmax,[-1])
        self.lable_=tf.argmax(self.y_, 1)
        self.rst = tf.cast(self.correct_prediction, "float")
        self.accuracy = tf.reduce_mean(self.rst)


if __name__=='__main__':
    recingnitionnum=RecingnitionNum()
    recingnitionnum.forward()
    recingnitionnum.backward()

    init = tf.global_variables_initializer()

    train = train_shuffle_batch(train_filename, [64, 64, 3], 128)
    test = test_shuffle_batch(test_filename, [64, 64, 3],1)
    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()      # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        sess.run(init)
        x=[]
        y=[]

        saver.restore(sess,"modle4_save/train5.dpk")
        yy=0
        flag = 0
        flag1 = 0
        for ii in range(10000):
            # iamge,labes=sess.run(train)
            # nm_iamge=iamge/255-0.5
            # trans = np.reshape(labes, [-1]).tolist()
            # lables = one_hot(10, trans)
            #
            # out_arg,_,loss,out,ac=sess.run([recingnitionnum.out_argmax ,recingnitionnum.optimizer,
            #                                         recingnitionnum.loss,recingnitionnum.out,recingnitionnum.accuracy],feed_dict={recingnitionnum.x:nm_iamge,
            #                                                                                  recingnitionnum.y_:lables})
            #
            # x.append(ii)
            # y.append(loss)
            # plt.plot(x,y,'red')
            # plt.pause(0.01)
            # plt.clf()
            # print("第 {} 次误差是 {}  ,accuracy :{} ".format(ii,loss,ac))
            #
            # #塞选训练集识别错误的数据
            # train_out=out_arg.tolist()
            # train_label=labes.tolist()
            #
            # for index in range(len(train_out)):
            #     if train_out[index]==train_label[index][0]:
            #         pass
            #     else:
            #         # scipy.misc.imsave('/Users/wywy/Desktop/train_bad/{}_{}_{}.jpg'.format(flag1,train_label[index][0],train_out[index]),iamge[index])
            #         flag1+=1


            if ii%10==0:
                test_img, test_lable = sess.run(test)
                nm_test_img = test_img / 255 - 0.5
                test_trans = np.reshape(test_lable, [-1]).tolist()
                test_lables = one_hot(10, test_trans)


                test_lable11 = np.reshape(test_lable, [-1]).tolist()
                loss1, out1,ac1,out_argmax,lable_ = sess.run([recingnitionnum.loss, recingnitionnum.out,recingnitionnum.accuracy,recingnitionnum.out_argmax,recingnitionnum.lable_],
                                        feed_dict={recingnitionnum.x: nm_test_img,
                                                   recingnitionnum.y_: test_lables})

                # graph_def = tf.get_default_graph().as_graph_def()
                # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                #                                                                 ['output'])
                #
                # with tf.gfile.GFile("./number_modle.pb", 'wb') as f:
                #     f.write(output_graph_def.SerializeToString())



                frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                               ['output'])  # 这里 ['output']是输出tensor的名字
                tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [recingnitionnum.x], [recingnitionnum.output
                                                                                                   ])  # 这里[input], [out]这里分别是输入tensor或者输出tensor的集合,是变量实体不是名字
                open("number_model5.tflite", "wb").write(tflite_model)
                #塞选测试集未识别准确数据
                # test_out=out_argmax.tolist()
                # test_label=lable_.tolist()
                #
                # for index1 in range(len(test_out)):
                #     if test_out[index1] == test_label[index1]:
                #         pass
                #     else:
                #         # scipy.misc.imsave(
                #         #     # '/Users/wywy/Desktop/test_bad/{}_{}_{}.jpg'.format(flag,test_label[index1], test_out[index1]),
                #         #     test_img[index1])
                #         flag+=1



                # print("测试第 {} 次误差是: {} 精度是: {}输出是   {}，输入时 {} ".format(ii, loss1,ac1,out_argmax[0],lable_[0]))
                # saver.save(sess,"modle4_save/train5.dpk")
                # saver.save(sess, "modle_save/RecingnitionNum.dpk")
                # img=test2_img[0]
                # plt.imshow(img)
                # plt.show()
                # plt.clf()












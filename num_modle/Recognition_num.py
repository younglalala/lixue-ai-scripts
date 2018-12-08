import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

from skimage import io

import matplotlib.image as mpimg
from num_modle.onehott import *
from num_modle.recognition_sample import *
from num_modle.test_sample import *
import tempfile
import subprocess
import cv2
tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess

class RecingnitionNum:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='input')
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 11])
        # self.dp=tf.placeholder(dtype=tf.float32)

        self.conv1_w = tf.Variable(tf.truncated_normal([3, 3, 1, 16], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        self.conv1_b = tf.Variable(tf.zeros([16]))
        # self.conv1_dw = tf.Variable(tf.truncated_normal([3, 3, 16, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        # self.conv1_db = tf.Variable(tf.zeros([16]))

        self.conv2_w = tf.Variable(tf.random_normal([3, 3, 16, 32], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        self.conv2_b = tf.Variable(tf.zeros([32]))
        # self.conv2_dw = tf.Variable(tf.truncated_normal([1, 1, 32, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 32)))
        # self.conv2_db = tf.Variable(tf.zeros([32]))

        # self.conv3_w = tf.Variable(tf.random_normal([3, 3, 32, 64], dtype=tf.float32, stddev=tf.sqrt(1 / 64)))
        # self.conv3_b = tf.Variable(tf.zeros([64]))

        self.fc1_w = tf.Variable(tf.truncated_normal([7 * 7 * 32, 128], dtype=tf.float32, stddev=tf.sqrt(1 / 256)))
        self.fc1_b = tf.Variable(tf.zeros([128]))

        self.fc3_w = tf.Variable(tf.truncated_normal([128, 11], dtype=tf.float32, stddev=tf.sqrt(1 / 11)))
        self.fc3_b = tf.Variable(tf.zeros([11]))

    def forward(self):
        self.conv1 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(self.x, self.conv1_w, strides=[1, 1, 1, 1],
                                                                           padding='SAME') + self.conv1_b))  # hape=(?, 30, 27, 10)
        self.pool1=tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        # self.conv2w = tf.nn.relu(tf.layers.batch_normalization(
        #     tf.nn.depthwise_conv2d(self.conv1, self.conv1_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv1_db))

        self.conv2 = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.pool1, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b))

        self.pool2=tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        # self.conv2w = tf.nn.relu(
        #     tf.nn.depthwise_conv2d(self.conv2, self.conv2_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_db)
        # self.conv3 = tf.nn.relu(tf.nn.conv2d(self.conv2w, self.conv3_w, strides=[1, 1, 1, 1],
        #                                      padding='SAME') + self.conv3_b)  # (?, 7, 7, 64)
        #
        # self.pool = tf.nn.avg_pool(self.conv3, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='SAME')

        flat = tf.reshape(self.pool2, [-1, 7 * 7 * 32])

        self.fc1 = tf.nn.relu(tf.matmul(flat, self.fc1_w) + self.fc1_b)
        # self.fc1=tf.nn.dropout(self.fc1,keep_prob=self.dp)



        self.out = tf.matmul(self.fc1, self.fc3_w) + self.fc3_b
        self.output=tf.reshape(self.out,[-1,11],name='output')

    def backward(self):


        self.cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.output)
                                         )
        # global_step = tf.Variable(0, trainable=False)
        # self.learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.96, staircase=True)


        self.optimizer=tf.train.AdamOptimizer(0.00002).minimize(self.cross_entropy)

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

    train = train_shuffle_batch(train_filename, [28, 28, 1], 128)
    test = test_shuffle_batch(test_filename, [28, 28, 1],1)
    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()      # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        sess.run(init)
        x=[]
        y=[]

        saver.restore(sess,"modle_save/m1train.dpk")
        yy=0
        flag = 0
        flag1 = 0
        for ii in range(10000):
            # iamge,labes,train_name=sess.run(train)
            # nm_iamge=iamge/255-0.5
            # trans = np.reshape(labes, [-1]).tolist()
            # lables = one_hot(10, trans)
            #
            # out_arg,_,loss,out,ac=sess.run([recingnitionnum.out_argmax ,recingnitionnum.optimizer,
            #                                         recingnitionnum.cross_entropy,recingnitionnum.out,recingnitionnum.accuracy],feed_dict={recingnitionnum.x:nm_iamge,
            #                                                                                  recingnitionnum.y_:lables,recingnitionnum.dp:0.7})
            #
            # # x.append(ii)
            # # y.append(loss)
            # # plt.plot(x,y,'red')
            # # plt.pause(0.1)
            # # plt.clf()
            # print("第 {} 次误差是 {}jingdu{} ".format(ii,loss,ac))
            # #
            # # # #塞选训练集识别错误的数据
            # train_out=out_arg.tolist()
            # train_label=labes.tolist()
            #
            # for index in range(len(train_out)):
            #     if train_out[index]==train_label[index][0]:
            #         pass
            #     else:
            #         file_name=bytes.decode(train_name[index])
            #         file_name=file_name.split('/')[-1]
            #         image = np.expand_dims(iamge[index], axis=2)
            #         image = np.concatenate((image, image, image), axis=2)
            #         image = image.reshape([28, 28, 3])
            #         scipy.misc.imsave('/Users/wywy/Desktop/train_e/{}'.format(file_name),image)
            #         flag1+=1


            if ii%500==0:


                # test_img=Image.open ('/Users/wywy/Desktop/xxx/19.jpg').convert('L')
                # test_img=np.array(test_img)
                #
                # test_img=test_img.reshape([-1,28,28,1])
                # test_img=test_img / 255 - 0.5
                # outt=sess.run(recingnitionnum.out_argmax,feed_dict={recingnitionnum.x:test_img,recingnitionnum.dp:1.})
                # print(outt)

                test_img, test_lable,test_name = sess.run(test)
                nm_test_img = test_img/255-0.5
                test_trans = np.reshape(test_lable, [-1]).tolist()
                test_lables = one_hot(10, test_trans)


                test_lable11 = np.reshape(test_lable, [-1]).tolist()
                loss1, out1,ac1,out_argmax,lable_ = sess.run([recingnitionnum.cross_entropy, recingnitionnum.out,recingnitionnum.accuracy,recingnitionnum.out_argmax,recingnitionnum.lable_],
                                        feed_dict={recingnitionnum.x: nm_test_img,
                                                   recingnitionnum.y_: test_lables})

                graph_def = tf.get_default_graph().as_graph_def()
                output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                                                                                ['output'])

                with tf.gfile.GFile("./number_modle1.pb", 'wb') as f:
                    f.write(output_graph_def.SerializeToString())

                # frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                #                                                                ['output'])  # 这里 ['output']是输出tensor的名字
                # tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [recingnitionnum.x],
                #                                             [recingnitionnum.output
                #                                              ])  # 这里[input], [out]这里分别是输入tensor或者输出tensor的集合,是变量实体不是名字
                # open("number_model6.tflite", "wb").write(tflite_model)
                #塞选测试集未识别准确数据
                # test_out=out_argmax.tolist()
                # test_label=lable_.tolist()
                #
                # for index1 in range(len(test_out)):
                #     if test_out[index1] == test_label[index1]:
                #         pass
                #     else:
                #         test_n=bytes.decode(test_name[index1])
                #         test_n=test_n.split('/')[-1]
                #         image = np.expand_dims(test_img[index1], axis=2)
                #         image = np.concatenate((image, image, image), axis=2)
                #         image = image.reshape([28, 28, 3])
                #         scipy.misc.imsave(
                #             '/Users/wywy/Desktop/test_e/{}'.format(test_n),
                #             image)
                #         flag+=1
                #
                #
                #
                print("测试第 {} 次误差是: {} 精度是: {}输出是   {}，输入时 {} ".format(ii, loss1,ac1,out_argmax[0],lable_[0]))
                # saver.save(sess,"modle_save/m1train.dpk")
                # saver.save(sess, "modle_save/RecingnitionNum.dpk")
                # img=test2_img[0]
                # plt.imshow(img)
                # plt.show()
                # plt.clf()












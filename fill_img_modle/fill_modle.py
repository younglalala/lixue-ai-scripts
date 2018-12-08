import scipy.misc
import cv2
import time
from fill_img_modle.onehott import *
from fill_img_modle.fill_sample import *
from fill_img_modle.fill_test_sample import *

import tempfile
import subprocess
tf.contrib.lite.tempfile = tempfile
tf.contrib.lite.subprocess = subprocess

strat=time.clock()
batch_size=100
class RecingnitionNum:
    def __init__(self):
        self.x=tf.placeholder(dtype=tf.float32,shape=[None,168,32,3],name='input')
        self.y_=tf.placeholder(dtype=tf.float32,shape=[None,11])
        # self.dp=tf.placeholder(dtype=tf.float32)
        # self.is_train = tf.placeholder(tf.bool)

        self.conv1_w=tf.Variable(tf.random_normal([3,3,3,6],dtype=tf.float32,stddev=tf.sqrt(1/6)),name='conv1_w')
        self.conv1_b=tf.Variable(tf.zeros([6]),name='conv1_b')
        self.conv1_dw=tf.Variable(tf.random_normal([3,3,6,1],dtype=tf.float32,stddev=tf.sqrt(1/6)))
        self.conv1_db=tf.Variable(tf.zeros([6]))

        # self.conv2_w=tf.Variable(tf.random_normal([3,3,8,16],dtype=tf.float32,stddev=tf.sqrt(1/16)),name='conv3_w')
        # self.conv2_b=tf.Variable(tf.zeros([16]),name='conv3_b')
        # self.conv2_dw = tf.Variable(tf.random_normal([3, 3, 16, 1], dtype=tf.float32, stddev=tf.sqrt(1 / 16)))
        # self.conv2_db = tf.Variable(tf.zeros([16]))


        self.fc1_w=tf.Variable(tf.random_normal([11*2*6,12],dtype=tf.float32,stddev=tf.sqrt(1/12)),name='fc1_w')
        self.fc1_b=tf.Variable(tf.zeros([12]),name='fc1_b')


        self.fc4_w=tf.Variable(tf.random_normal([12,11],dtype=tf.float32,stddev=tf.sqrt(1/10)),name='fc4_w')
        self.fc4_b=tf.Variable(tf.zeros([11]),name='fc4_b')

    def forward(self):

        self.conv1=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.conv2d(self.x,self.conv1_w,strides=[1,2,2,1],padding='SAME')+self.conv1_b) ) #shape=(?, 30, 27, 10)
        self.pool1=tf.nn.avg_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        self.conv1d=tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.depthwise_conv2d(self.pool1,self.conv1_dw,strides=[1,2,2,1],padding='SAME')+self.conv1_db
        ))
        self.pool = tf.nn.max_pool(self.conv1d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # self.conv2 = tf.nn.relu(tf.layers.batch_normalization(
        #     tf.nn.conv2d(self.pool, self.conv2_w, strides=[1, 2, 2, 1],
        #                  padding='SAME') + self.conv2_b))  # shape=(?, 30, 27, 10)
        #
        # self.conv2d = tf.nn.relu(tf.layers.batch_normalization(
        #     tf.nn.depthwise_conv2d(self.conv2, self.conv2_dw, strides=[1, 2, 2, 1], padding='SAME') + self.conv2_db
        # # ))
        # self.pool1 = tf.nn.max_pool(self.conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #
        # self.conv3 = tf.nn.relu(tf.layers.batch_normalization(
        #     tf.nn.conv2d(self.pool1, self.conv4_w, strides=[1, 2, 2, 1],
        #                  padding='SAME') + self.conv4_b))  # shape=(?, 30, 27, 10)

        print(self.pool)
        #
        flat=tf.reshape(self.pool,[-1,11*2*6])
        self.fc1=tf.nn.relu(tf.matmul(flat,self.fc1_w)+self.fc1_b)
        # self.fc1_d= tf.nn.dropout(self.fc1, keep_prob=self.dp)


        self.out=tf.matmul(self.fc1,self.fc4_w)+self.fc4_b
        self.output=tf.reshape(self.out,[-1,11],name='output')


    def backward(self):
        self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.output))
                                 # 0.0001 * tf.reduce_sum(tf.pow(self.conv1_w, 2))+ 0.0001 * tf.reduce_sum(tf.pow(self.conv1_dw, 2))
                                 # +0.0001 * tf.reduce_sum(tf.pow(self.fc1_w, 2)))
        self.optimizer=tf.train.AdamOptimizer(0.0006).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y_, 1))
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

    train = train_shuffle_batch(train_filename, [168, 32, 3], 256)
    test = test_shuffle_batch(test_filename, [168, 32, 3], 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        sess.run(init)
        x=[]
        y=[]

        saver.restore(sess,"modle_save/fill_num_img3.dpk")
        flag=0
        for ii in range(10000):
            # 训练集部分
            # train_iamge,train_labes=sess.run(train)
            # nm_iamge=train_iamge/255-0.5
            #
            # trans=np.reshape(train_labes,[-1]).tolist()
            # lables=one_hot(10,trans)
            #
            # _,loss,out,ac,out_arg,yy=sess.run([recingnitionnum.optimizer,recingnitionnum.loss,recingnitionnum.out,recingnitionnum.accuracy,recingnitionnum.out_argmax1,recingnitionnum.lable_],feed_dict={recingnitionnum.x:nm_iamge,
            #                                                                                      recingnitionnum.y_:lables})
            #
            # # print(out)
            # # elapsed = (time.clock() - strat)
            # #
            # # print('Time:{}'.format(elapsed))
            # x.append(ii)
            # y.append(loss)
            # print("第 {} 次误差是 {}，精度：{} ".format(ii,loss,ac))
            # plt.plot(x,y,'red')
            # plt.pause(0.01)
            # plt.clf()
            # train_c=0
            # for xx in range(len(out_arg)):
            #     if out_arg[xx]==yy[xx]:
            #         pass
            #     else:
            #         scipy.misc.imsave('/Users/wywy/Desktop/train_bad/{}_out{}_lable{}.jpg'.format(train_c,out_arg[xx],yy[xx]),
            #                           train_iamge[xx])
            #         train_c+=1


            #测试集部分。
            if ii%10==0:
                # all_img=[]
                # for file in os.listdir('/Users/wywy/Desktop/test'):
                #     if file=='.DS_Store':
                #         os.remove('/Users/wywy/Desktop/test'+'/'+file)
                #     else:
                #         test_img=cv2.imread('/Users/wywy/Desktop/test'+'/'+file)
                #         all_img.append(test_img)



                test_img=cv2.imread('/Users/wywy/Desktop/id_error_1.jpg')
                test_img=test_img/255-0.5
                test_img=test_img.reshape([1,168,32,3])
                test_out=sess.run(recingnitionnum.out_argmax,feed_dict={recingnitionnum.x:np.array(test_img)})
                print(test_out)
                # test_out=test_out.tolist()
                # count=0
                # for index in range(len(test_out)):
                #     scipy.misc.imsave('/Users/wywy/Desktop/test1'+'/'+str(count)+'_'+str(test_out[index])+'.jpg',np.array(all_img)[index])
                #     count+=1



                # test_img, test_lable = sess.run(test)
                # nm_test_img = test_img / 255-0.5
                # test_trans = np.reshape(test_lable, [-1]).tolist()
                # test_lables = one_hot(10, test_trans)
                # loss1, out1,ac1,out_argmax,lable_ = sess.run([recingnitionnum.loss, recingnitionnum.out,recingnitionnum.accuracy,recingnitionnum.out_argmax,recingnitionnum.lable_],
                #                         feed_dict={recingnitionnum.x: nm_test_img,
                #                                    recingnitionnum.y_: test_lables})

                # for xx1 in range(len(out_argmax)):
                #     if out_argmax[xx1] == lable_[xx1]:
                #         pass
                #     else:
                #         scipy.misc.imsave(
                #             '/Users/wywy/Desktop/test_bad/{}_out{}_lable{}.jpg'.format(test_name[xx1][0], out_argmax[xx1],
                #                                                                        lable_[xx1]),
                #             test_img[xx1])



            #     out_argmax11 = out_argmax.tolist()
            #     lables11 = np.reshape(test_lable, [-1]).tolist()
            #
            #     # graph_def = tf.get_default_graph().as_graph_def()
            #     # output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
            #     #                                                                 ['output'])
            #     # tflite_model = tf.contrib.lite.toco_convert(sess.graph_def, [recingnitionnum.x], [recingnitionnum.y_])
            #     # open("fill_img_model.tflite", "wb").write(tflite_model)
            #     #
            #     # with tf.gfile.GFile("./fill_modle.pb", 'wb') as f:
            #     #     f.write(output_graph_def.SerializeToString())
            #
            #     #
            #
            #     #
            #     frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
            #                                                                    ['output'])  # 这里 ['output']是输出tensor的名字
            #     tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, [recingnitionnum.x], [recingnitionnum.output
            #         ])  # 这里[input], [out]这里分别是输入tensor或者输出tensor的集合,是变量实体不是名字
            #     open("fill_img_model3.tflite", "wb").write(tflite_model)

            graph_def = tf.get_default_graph().as_graph_def()
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def,
                                                                            ['output'])

            with tf.gfile.GFile("./fill_modle1.pb", 'wb') as f:
                f.write(output_graph_def.SerializeToString())

            # #
            #
            #
            #
            #     # 存储测试数据中标注有错误的数据方便观察
            #     # 把测试集中分类错误的数据单独装在'/Users/wywy/Desktop/bad文件下面
            #     # xx=-1
            #     # for index in  out_argmax11:
            #     #     xx += 1
            #     #     if index!=lables11[xx]:
            #     #         pass
            #     #         flag+=1
            #     #         scipy.misc.imsave('/Users/wywy/Desktop/bad/{}_lable{}_out{}.jpg'.format(flag, lables11[xx], index),test_img[xx])
            #     print("测试第 {} 次误差是: {} 精度是: {}输出是   {}，输入时 {} ".format(ii, loss1,ac1,out_argmax[0],lable_[0]))
            #     # print('xxx')
            #     saver.save(sess, "modle_save/fill_num_img3.dpk")
            #











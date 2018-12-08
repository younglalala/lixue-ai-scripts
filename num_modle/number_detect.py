import numpy as np
import  os
import scipy.misc
import tensorflow as tf
import time
from num_modle.onehott import *
from num_modle.test_sample import *



#输入数据要求：1.输入ID区域图片要求（64，64，3），数据读成numpy格式即可传入网络进行识别
#输出数据格式：1.输出结果为每张图片的识别结果。输出结果是numpy格式
strat=time.clock()
class NumDetect:
    def __init__(self):
        pass
    def detect(self,batch_image):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            output_graph_path = "./number_modle.pb"

            with open(output_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                input_x = sess.graph.get_tensor_by_name("input:0")
                input_dp = sess.graph.get_tensor_by_name("dp:0")   #dropout
                output = sess.graph.get_tensor_by_name("output:0")

                cls = sess.run(output, feed_dict={input_x: batch_image,input_dp:1.0})

                return cls

    #这部分根据实际情况来删除（目前是方便测试），此部分为tfrecords处理好了的数据，可自行读取
    def get_batch(self,batch_size):
        test = test_shuffle_batch(test_filename, [64, 64, 3], batch_size)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            test_img, test_lable = sess.run(test)
            nm_test_img = test_img
            test_trans = np.reshape(test_lable, [-1]).tolist()
            test_lables = one_hot(10, test_trans)
            return nm_test_img,test_lables


num_detect=NumDetect()
img,lable=num_detect.get_batch(100)
input_image=img/255-0.5


cls=num_detect.detect(input_image)

#
cls=cls.reshape([-1]).tolist()
elapsed = (time.clock() - strat)
#
print('Time:{}'.format(elapsed))
print(cls)    #自行对分类进行输出标记


save_path='/Users/wywy/Desktop/number_img'
for index in range(len(img)):
    scipy.misc.imsave(save_path+'/'+str(index)+'_'+str(cls[index])+'.jpg',img[index])








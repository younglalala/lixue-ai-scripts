import numpy as np
import  os
import scipy.misc
import tensorflow as tf
import cv2
import time
from fill_img_modle.onehott import *
from fill_img_modle.fill_test_sample import *


#输入数据要求：1.输入ID区域图片要求（64，64，3），数据读成numpy格式即可传入网络进行识别
#输出数据格式：1.输出结果为每张图片的识别结果。输出结果是numpy格式
strat=time.clock()
class FillDetect:
    def __init__(self):
        pass
    def detect(self,batch_image):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            output_graph_path = "./fill_modle.pb"

            with open(output_graph_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                input_x = sess.graph.get_tensor_by_name("input:0")
                output = sess.graph.get_tensor_by_name("output:0")


                cls = sess.run(output, feed_dict={input_x: batch_image})

                return cls

    #这部分根据实际情况来删除（目前是方便测试），此部分为tfrecords处理好了的数据，可自行读取
    def get_batch(self,batch_size):
        test = test_shuffle_batch(test_filename, [64, 64, 3], batch_size)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            test_img, test_lable = sess.run(test)
            nm_test_img = test_img / 255 - 0.5
            test_trans = np.reshape(test_lable, [-1]).tolist()
            test_lables = one_hot(10, test_trans)
            return nm_test_img,test_lables


fill_detect=FillDetect()
# img,lable=fill_detect.get_batch(1000)
img=cv2.imread('/Users/wywy/Desktop/label_1.jpg')/255-0.5
# print(img.shape)
img=img.reshape([1,318,64,3])
# print(img)
# f = open("./text.txt",'w')
# f.write(img)
# f.close()


cls=fill_detect.detect(img)

cls=cls.reshape([-1]).tolist()
elapsed = (time.clock() - strat)

print('Time:{}'.format(elapsed))
print(cls)    #自行对分类进行输出标记








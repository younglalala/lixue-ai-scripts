import random
import os

import tensorflow as tf
import numpy as np
import matplotlib.image as iming
from PIL import Image
import matplotlib.pyplot as plt



lable_all='/Users/wywy/Desktop/all_train_fill'
train_filename = './train.tfrecords'  # 输出文件地址


def saver_lables(lable_all,train_filename):
    writer = tf.python_io.TFRecordWriter(train_filename)
    all_path=[]
    for i in os.listdir(lable_all):
        if i == '.DS_Store':
            pass
        else:
            all_path.append(lable_all+'/'+i)
    random.shuffle(all_path)
    for j in all_path:
        lable=j.split('/')[-1].split('.')[0].split('_')[-1]
        # name=float(j.split('/')[-1].split('.')[0].split('_')[0])
        if lable=='X' or lable=='10':
            lable=float(10)
        else:
            lable=float(lable)
        # print(lable)
        img=Image.open(j)
        img=img.resize((32,168),Image.ANTIALIAS)
        image = img.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
                'lables': tf.train.Feature(float_list=tf.train.FloatList(value=[lable])),
                # 'file_name': tf.train.Feature(float_list=tf.train.FloatList(value=[name,lable1])),
                'images': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
            }))
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


def read_data_for_file(file, capacity,image_size):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([file], num_epochs=None,shuffle=True, capacity=capacity)

    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)

    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'lables': tf.FixedLenFeature([], tf.float32),
            # 'file_name': tf.FixedLenFeature([2], tf.float32),
            'images':tf.FixedLenFeature([], tf.string)
        }
    )
    img = tf.decode_raw(features['images'], tf.uint8)
    img=tf.reshape(img,image_size)

    img = tf.cast(img, tf.int32)
    lables = tf.cast(features['lables'], tf.int32)
    lables=tf.reshape(lables,[1])

    # file_name=tf.cast(features['file_name'], tf.int32)


    return img, lables


def train_shuffle_batch(train_file_path, image_size,batch_size, capacity=700, num_threads=3):
    images, lables = read_data_for_file(train_file_path, 1000,image_size)

    images_,  lables_ = tf.train.shuffle_batch([images,lables], batch_size=batch_size, capacity=capacity,
                                               min_after_dequeue=100,
                                               num_threads=num_threads)
    return images_, lables_



init = tf.global_variables_initializer()
if __name__=='__main__':
    saver_lables(lable_all,train_filename)
    a=train_shuffle_batch(train_filename,[168,32,3],100)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord,sess=sess)

        sess.run(init)
        for ii in range(100):
            aa, bb = sess.run(a)
            print(bb[0])
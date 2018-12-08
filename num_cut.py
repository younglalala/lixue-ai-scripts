# coding=utf-8
import operator
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc
import os

import cv2
import numpy as np
from PIL import  Image
from tornado.options import options
import cv2
# exam_id=957
# img_path='/Users/wywy/Desktop/all_img/{}'.format(exam_id)   #7
# save_img='/Users/wywy/Desktop/ccc'.format(exam_id)
# count=7843
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img_size=img.size
#
#         aa=img.crop((0,55,img_size[0],60+60))
#         aa.save(save_img+'/'+str(count)+'_'+file)
#         count+=1
# print(count)

# img_path='/Users/wywy/Desktop/数字整串识别部分/train_num'
# save_path='/Users/wywy/Desktop/train_num'
#
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.listdir(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         img=Image.open(img_path+'/'+file)
#         w,h=img.szie
#         x1=0
#         y1=0
#         x2=
#         # for  index in range(5):
#         #     img.crop(())












exam_id=954
img_path='/Users/wywy/Desktop/数据集/答题卡ID区域数据集/all_id_img/id_img1'.format(exam_id)   #7
save_img='/Users/wywy/Desktop/test'.format(exam_id)

# save_path1='/Users/wywy/Desktop/test_num'

class NumParser:
    def __init__(self, img):
        self.ID_NUM = 5
        self.img = img
        self.height, self.width = self.img.shape[:2]
        self.iamge_w=self.width/self.ID_NUM
        self.image_h=330.
    def get_img(self):
        all_img=[]
        for i in range(self.ID_NUM):
            x=int(i*self.iamge_w)+2
            y=int(100)
            w=int(np.ceil(self.iamge_w))-6

            h=int(self.image_h)
            # image_lable.append((x,y,w,h))
            img = self.img[y: y + h, x: x + w]
            all_img.append(img)
        return all_img
if __name__=='__main__':
    count=0
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            name=list(file.split('.')[0].split('_')[-1])

            x = cv2.imread(img_path+'/'+file)
            numparser=NumParser(x)
            all_img=numparser.get_img()
            xx=0
            for img in all_img:
                # print(save_img+'/'+str(count)+'_'+str(name[xx])+'.jpg')
                # print(save_img+'/'+str(xx1)+'_'+str(name[xx])+'.jpg')
                scipy.misc.imsave(save_img+'/'+str(count)+'_'+str(name[xx])+'.jpg', img)
                xx+=1
                count+=1
    print(count)

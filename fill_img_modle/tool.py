
import os
import numpy as np
import cv2 as cv
from PIL import Image

# img_path='/Users/wywy/Desktop/训练完成模型/ID区域识别填涂部分/最后一次模型数据集（ID填涂部分）/train_cls'
# img1_path='/Users/wywy/Desktop/test_bad'
# for file in os .listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=cv.imread(img_path+'/'+file)
#         for file1 in os.listdir(img1_path):
#             if file1=='.DS_Store':
#                 os.remove(img1_path+'/'+file1)
#             else:
#                 img1=cv.imread(img1_path+'/'+file1)
#                 if (img==img1).all():
#                     print(img,'-----------')
#                     print(img1,'==========')


img_path='/Users/wywy/Desktop/test'
count=0
for file in os.listdir(img_path):
    if file=='.DS_Store':
        os.remove(img_path+'/'+file)
    else:
        img=Image.open(img_path+'/'+file)
        out=img.resize((32,168),Image.ANTIALIAS)
        out.save(img_path+'/'+file)
        count+=1

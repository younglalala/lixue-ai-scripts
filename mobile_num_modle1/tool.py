import os
from PIL import Image
import cv2
import scipy.misc
import random

import numpy as np


# img_path='/Users/wywy/Desktop/test_num_img'

def resize_picture(img_path,save_path):
    for i in os.listdir(img_path):
        if i=='.DS_Store':
            os.remove(img_path+'/'+i)
        else:
            im=Image.open(img_path+'/'+i)

            out=im.resize((64,64),Image.ANTIALIAS)
            out.save(save_path+'/'+i)
            # print(save_path+'/'+i)
# resize_picture(img_path,img_path)


#删除错误数据
# img_path='/Users/wywy/Desktop/num_train_flase'
# img_path1='/Users/wywy/Desktop/train_num_img'
# img_path='/Users/wywy/Desktop/num_test_false'
# img_path1='/Users/wywy/Desktop/test_num_img'


# img_path='/Users/wywy/Desktop/train_bad'
# img_path1='/Users/wywy/Desktop/train_fill_img'





def remove_flase(img_path,img_path1):
    img_names=[]
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            name=file.split('.')[0].split('_')
            label=list(name[-1])[-1]
            img_name=name[0]+'_'+label

            img_names.append(img_name)
    img_names=list(set(img_names))
    cc=0
    for names in img_names:
        os.remove(img_path1+'/'+names+'.jpg')
        cc+=1
        print('remove   {}.jpg   ok'.format(names))
    print(cc)

# remove_flase(img_path,img_path1)
# count=541510
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=list(file.split('.')[0].split('_')[-1])[-1]
#         img=Image.open(img_path+'/'+file)
#         img.save(img_path1+'/'+str(count)+'_'+name+'.jpg')
#         # print(img_path1+'/fill'+str(count)+'_'+name+'.jpg')
#         count+=1
# print(count)

# for file in os.listdir(img_path1):
#     if file=='.DS_Store':
#         os.remove(img_path1+'/'+file)
#     else:
#         name=list(file.split('.')[0].split('_')[0])[0]
#         if name=='f':
#             os.remove(img_path1+'/'+file)
#             print(file)









#分类塞选
# xx=9
# img_path='/Users/wywy/Desktop/train_fill_img'
# save_path='/Users/wywy/Desktop/train_cls/{}'.format(xx)
#
# img_path='/Users/wywy/Desktop/test_fill_img'
# save_path='/Users/wywy/Desktop/test_cls/{}'.format(xx)


def clas(img_path,save_path,num):

    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            name=file.split('.')[0].split('_')[-1]
            img=Image.open(img_path+'/'+file)
            if name==str(num):
                img.save(save_path+'/'+file)

# clas(img_path,save_path,xx)



###remove img
# xx=9
# img_path='/Users/wywy/Desktop/train_分类/{}'.format(xx)
# save_path='/Users/wywy/Desktop/train_num_img'

# img_path='/Users/wywy/Desktop/test_分类/{}'.format(xx)
# save_path='/Users/wywy/Desktop/test_num_img'
# idd=955
# img_path='/Users/wywy/Desktop/new_fill_img/{}'.format(idd)
# save_path='/Users/wywy/Desktop/train_fill_img'
# save_path1='/Users/wywy/Desktop/test_fill_img'

# img_path='/Users/wywy/Desktop/train_num_img'
# img_path1='/Users/wywy/Desktop/train_num_aug'
# save_path='/Users/wywy/Desktop/all_train_numimg'

# img_path='/Users/wywy/Desktop/test_num_img'
# img_path1='/Users/wywy/Desktop/test_num_aug'
# save_path='/Users/wywy/Desktop/all_test_numimg'

def path_remove(img_path,save_path):
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(img_path+'/'+file)
            img.save(save_path+'/'+file)

    print('path remove finish....')
# path_remove(img_path1,save_path)


#
# img_path='/Users/wywy/Desktop/train_num_img'
# save_path='/Users/wywy/Desktop/train_num_aug'
#
# img_path='/Users/wywy/Desktop/train_fill_img'
# save_path='/Users/wywy/Desktop/train_fill_img_aug'
img_path='/Users/wywy/Desktop/train_num_aug3'
save_path='/Users/wywy/Desktop/all_train_num'



def img_aug(img_path,save_path):
    all_path=[]
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            all_path.append(img_path+'/'+file)
    random.shuffle(all_path)
    count=10000
    for i in range(2000):
        index=random.randint(0,len(all_path)-1)
        img=cv2.imread(all_path[index])
        name=all_path[index].split('.')[0].split('_')[-1]
        img1 = img
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img1[i, j] =img1[i, j][0]*0.95,img1[i, j][1]*0.95,img1[i, j][2]*0.95
        scipy.misc.imsave(save_path+'/3aug'+str(count)+'_'+name+'.jpg', img1)
        # print(save_path+'/aug'+str(count)+'_'+name+'.jpg')
        count+=1
    print(count)

# img_aug(img_path,save_path)


#数据增强（增加噪点，镜像）
def salt(img_path,n,save_path):
    all_path=[]
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            all_path.append(img_path+'/'+file)
    random.shuffle(all_path)
    count = 30000
    for i in range(2000):
        index = random.randint(0, len(all_path) - 1)
        img = cv2.imread(all_path[index])
        img1 = img
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img1[i, j] = img1[i, j][0] * 0.95, img1[i, j][1] * 0.95, img1[i, j][2] * 0.95
        name = all_path[index].split('.')[0].split('_')[-1]

        for k in range(n):
            # 随机选择椒盐的坐标
            i = int(np.random.random() * img1.shape[1])
            j = int(np.random.random() * img1.shape[0])
            # 如果是灰度图
            if img1.ndim == 2:
                img1[j, i] = 255
                # 如果是RBG图片
            elif img1.ndim == 3:
                img1[j, i, 0] = 0
                img1[j, i, 1] = 0
                img1[j, i, 2] = 0
        scipy.misc.imsave(save_path+'/3aug'+str(count)+'_'+name+'.jpg', img1)
        count+=1
    print(count)
# salt(img_path,300,save_path)



# img_path='/Users/wywy/Desktop/test_cls/2'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         if name=='2':
#             pass
#         else:
#             os.remove(img_path+'/'+file)
#             print(file)

# img_path='/Users/wywy/Desktop/ok_test_num'
# img_path1='/Users/wywy/Desktop/test_num_aug'
# save_path='/Users/wywy/Desktop/all_test_num'
#
# # img_path='/Users/wywy/Desktop/ok_test'
# # img_path1='/Users/wywy/Desktop/ok_test_aug'
# # save_path='/Users/wywy/Desktop/all_test_fill'
# #


# img_path='/Users/wywy/Desktop/train_num_aug3'
# save_path='/Users/wywy/Desktop/ok_train_num'
#

def remove_image(img_path,save_path):
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(img_path+'/'+file)
            img.save(save_path+'/'+file)
# remove_image(img_path,save_path)

# img_path='/Users/wywy/Desktop/all_train_cls10'
# save_path='/Users/wywy/Desktop/all_train_num'
# count=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+str(count)+'_X.jpg')
#         # print(save_path+'/'+str(count)+'_X.jpg')
#         count+=1
# print(count)


# img_path='/Users/wywy/Desktop/output'
# save_path='/Users/wywy/Desktop/all_train_num'
# count=4000
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/aug0'+str(count)+'_'+name+'.jpg')
#         # print(save_path+'/aug0'+str(count)+'_'+name+'.jpg')
#         count+=1
# print(count)








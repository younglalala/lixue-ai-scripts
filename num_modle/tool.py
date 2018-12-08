import os
import cv2
import random
import scipy.misc
import  numpy as np
from PIL import Image,ImageDraw
# save_path='/Users/wywy/Desktop/part_img'
def resize_picture(img_path,save_path):
    for i in os.listdir(img_path):
        if i=='.DS_Store':
            os.remove(img_path+'/'+i)
        else:
            im=Image.open(img_path+'/'+i)

            out=im.resize((12,20),Image.ANTIALIAS)
            out.save(save_path+'/'+i)
# resize_picture(save_path,save_path)

# img_path='/Users/wywy/Desktop/train_bad'
# save_path='/Users/wywy/Desktop/img_aug'

def img_aug(img_path,save_path):
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=cv2.imread(img_path+'/'+file)
            img1=img
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img1[i,j][0]<150:
                        img1[i, j] =220,220,220
            scipy.misc.imsave(save_path+'/'+file, img)
# img_aug(img_path,save_path)



#改变文件路径

# img_path='/Users/wywy/Desktop/test_bad'
# save_path='/Users/wywy/Desktop/train_num_img'

def change_path(img_path,save_path):
    flag=250377
    for file in os.listdir(img_path):
        if file =='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            name=file.split('.')[0].split('_')[1]
            img=Image.open(img_path+'/'+file)
            # print(save_path+'/'+str(flag)+'_'+name+'.jpg')
            img.save(save_path+'/'+str(flag)+'_'+name+'.jpg')
            flag+=1
    print(flag)
# change_path(img_path,save_path)

#图像增加噪音
# img_path='/Users/wywy/Desktop/train_bad'
# save_path='/Users/wywy/Desktop/img_aug'
#
# def img_aug2(img_path,save_path):
#     for file in os.listdir(img_path):
#         if file=='.DS_Store':
#             os.remove(img_path+'/'+file)
#         else:
#             img=Image.open(img_path+'/'+file)
#             draw=ImageDraw.Draw(img)
#             draw.line((random.randint(0,64),random.randint(0,64),random.randint(0,64),random.randint(0,64)),fill=(0,0,0))
#             img.save(save_path+'/'+file)
# img_aug2(img_path,save_path)
# # #
# img_path='/Users/wywy/Desktop/train_bad'
# save_path='/Users/wywy/Desktop/img_aug'


# img_path1='/Users/wywy/Desktop/part_img'
# img_path2='/Users/wywy/Desktop/train_bad'
def img_aug3(img_path1,img_path2,save_path):
    all_paste=[]
    for file in os.listdir(img_path1):
        if file=='.DS_Store':
            os.remove(img_path1+'/'+file)
        else:
            img_paste=Image.open(img_path1+'/'+file)
            all_paste.append(img_paste)
    for file2 in os.listdir(img_path2):
        if file2 =='.DS_Store':
            os.remove(img_path2+'/'+file2)
        else:
            img2=Image.open(img_path2+'/'+file2)
            index=random.randint(len(all_paste))
            img2.paste(all_paste[index],)



#删除错误数据

# img_path='/Users/wywy/Desktop/test_bad'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[1:]
#         if name[0]=='1' and name[1]=='3':
#
#             os.remove(img_path+'/'+file)


# img1_path='/Users/wywy/Desktop/ob_aug'
# img_path='/Users/wywy/Desktop/7_c'
# save_path='/Users/wywy/Desktop/训练完成模型/客观题数据（单、多选）/train1_img'
# count=407008
# for file in os.listdir(img1_path):
#     if file == '.DS_Store':
#         os.remove(img1_path + '/' + file)
#     else:
#         img=Image.open(img1_path+'/'+file)
#         name=file.split('.')[0].split('_')[-1]
#         img.save(save_path+'/'+str(count)+'_'+name+'.jpg')
#         # print(save_path+'/'+str(count)+'_'+name+'.jpg')
#         count+=1
# print(count)




#
# img_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/test_cls/test_all'
# save_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/test_cls/aug2'
def img_aug_for_number(img_path,save_path):
    count=0
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            name=file.split('.')[0].split('_')[-1]

            img = cv2.imread(img_path + '/' + file)
            img1 = img
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img1[i, j] = img[i,j][0]*0.6,img[i,j][1]*0.6,img[i,j][2]*0.6

            scipy.misc.imsave(save_path + '/o2baug' + str(count)+'_'+name+'.jpg', img1)
            # print(save_path + '/obaug' + str(count)+'_'+name+'.jpg')
            count+=1
    print(count)

# img_aug_for_number(img_path,save_path)
#
# img_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/test_cls/aug2'
# save_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/test_cls/test_all'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+file)
# print('ok')







def number_aug1(img_path,save_path):
    count=0
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            name = file.split('.')[0].split('_')[-1]
            img = Image.open(img_path + '/' + file)
            crop_x1=random.randint(0,6)
            crop_y1=random.randint(0,6)
            crop_x2=random.randint(58,64)
            crop_y2=random.randint(58,64)
            region = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            region.save(save_path+'/numgug1_'+str(count)+'_'+name+'.jpg')
            # print(save_path+'/numgug1_'+str(count)+'_'+name+'.jpg')
            count+=1

# number_aug1(img_path,save_path)



#

# img_path='/Users/wywy/Desktop/cut1'
# img_path1='/Users/wywy/Desktop/cccc'
# save_path='/Users/wywy/Desktop/7_c'
def img_paste1(img_path,img_path1,save_path):
    all_paste_img=[]
    all_paste_name=[]
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=Image.open(img_path+'/'+file)
            name=file.split('.')[0].split('_')[-1]
            all_paste_name.append(name)
            img=img.resize((36,16),Image.ANTIALIAS)
            all_paste_img.append(img)
    count=96
    for file1 in os.listdir(img_path1):
        if file1=='.DS_Store':
            os.remove(img_path1+'/'+file1)
        else:
            img1=Image.open(img_path1+'/'+file1)
            index=random.randint(0,len(all_paste_name)-1)
            img1.paste(all_paste_img[index],(123,0))
            img1.save(save_path+'/'+str(count)+'_'+all_paste_name[index]+'.jpg')
            # print(save_path + '/' + str(count) + '_' + all_paste_name[index] + '.jpg')
            count+=1
    print(count)



# id=954
# img_path='/Users/wywy/Desktop/填涂部分正数据/{}'.format(id)
# # save_path='/Users/wywy/Desktop/train_fill'
# save_path='/Users/wywy/Desktop/test_fill'
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+file)

# cls=9
# save_path='/Users/wywy/Desktop/ok_test'
# img_path='/Users/wywy/Desktop/test_cls/{}'.format(cls)


# img_path='/Users/wywy/Desktop/test_fill'
# save_path='/Users/wywy/Desktop/test_cls'

# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+file)



#删除错误数据


# img_path='/Users/wywy/Desktop/train_bad'
# img_path1='/Users/wywy/Desktop/ok_train'
# all_filename=[]
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         num=file.split('.')[0].split('_')[0]
#         label=list(file.split('.')[0].split('_')[-1])[-1]
#         all_filename.append('{}_{}.jpg'.format(num,label))
# for r in list(set(all_filename)):
#     os.remove(img_path1+'/'+r)
#     print('remove  {}   ok'.format(r))


# img_path='/Users/wywy/Desktop/train_num'
# save_path='/Users/wywy/Desktop/test_num'
# count=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         if count<1700:
#             img=Image.open(img_path+'/'+file)
#             img.save(save_path+'/'+file)
#             os.remove(img_path+'/'+file)
#             count+=1
# print(count)
# cls=0
# # img_path='/Users/wywy/Desktop/test_num/{}'.format(id)
# img_path='/Users/wywy/Desktop/test_cls/{}'.format(cls)
# save_path='/Users/wywy/Desktop/ok_test_num'
#
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+file)
        # print(save_path+'/'+file)



# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         img=Image.open(img_path+'/'+file)
#         if name=='0':
#             img.save(save_path+'/0/'+file)
#         elif name=='1':
#             img.save(save_path + '/1/' + file)
#         elif name=='2':
#             img.save(save_path + '/2/' + file)
#         elif name=='3':
#             img.save(save_path + '/3/' + file)
#         elif name=='4':
#             img.save(save_path + '/4/' + file)
#         elif name=='5':
#             img.save(save_path + '/5/' + file)
#
#         elif name=='6':
#             img.save(save_path + '/6/' + file)
#
#         elif name=='7':
#             img.save(save_path + '/7/' + file)
#
#         elif name=='8':
#             img.save(save_path + '/8/' + file)
#         elif name == '9':
#             img.save(save_path + '/9/' + file)

# import os
# import scipy.misc
# import cv2
# from math import *
# import numpy as np
# # for file in os.listdir(img_path):
# #     if file=='.DS_Store':
# #         os.remove(img_path+'/'+file)
# #     else:
# #         label=file.split('_')[-1].split(',')
# #         x1,y1,x2=float(label[0]) ,float(label[1]),float(label[2])
# #         y2=float(label[-1].split('.')[0]+'.'+label[-1].split('.')[1])
# #         img=Image.open(img_path+'/'+file)
# #         drawObject = ImageDraw.Draw(img)
# #         drawObject.rectangle((x1,y1,x2,y2),outline = "red")
#         # img.show()
#
#
#
# # 旋转angle角度，缺失背景白色（255, 255, 255）填充
def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 0.75)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))

# #
img_path='/Users/wywy/Desktop/判断题数据/train_choice2'
save_path='/Users/wywy/Desktop/判断题数据/aug'

x=0
for file in os.listdir(img_path):
    if file=='.DS_Store':
        os.remove(img_path+'/'+file)
    else:
        angle=np.random.randint(30)
        img = cv2.imread(img_path + '/' + file)
        index=random.sample([-1,1],1)
        # print(file)
        name=file.split('.')[0].split('_')[-1]
        imgRotation = rotate_bound_white_bg(img, angle*int(index[0]))
        h,w,c=np.shape(imgRotation)
        # h_setoff,w_setoff=round( h/240,1),round( w/340,1)
        # print(h_setoff,w_setoff)
        # a=scipy.misc.imresize(imgRotation,(240,340))
        # print('/Users/wywy/Desktop/rotate/{}_{}_{}_.jpg'.format(x,h_setoff,w_setoff))
        scipy.misc.imsave('/Users/wywy/Desktop/判断题数据/aug/aug{}_{}.jpg'.format(x,name), imgRotation)
        x+=1
print(x)






# img_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/train_cls/1010'
# # img_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/train_cls/00'
# save_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/train_cls/train_all'
#
# count=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+file)
#         count+=1
# print(count)







# img_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/all_train_num'
# save_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/train_cls/1010'
#
# for file in os.listdir(img_path):
#     if file == '.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         name2=file.split('.')[0].split('_')[0]
#         if name=='X':
#             if 'a' in name2:
#                 pass
#             else:
#                 img=Image.open(img_path+'/'+file)
#                 crop_img=img.crop((3,3,img.size[0]-4,img.size[1]-4))
#                 crop_img.save(save_path+'/'+file)



#
# img_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/train_cls/00'
# save_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/train_cls/aug0'
#
#
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#
#         crop_img=img.crop((2,2,img.size[0]-5,img.size[1]-10))
#         crop_img.save(save_path+'/'+file)


# img_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/train_cls/train_all'
# img_path1='/Users/wywy/Desktop/train_e'
#
# count=0
# for file in os.listdir(img_path1):
#     if file=='.DS_Store':
#         os.remove(img_path1+'/'+file)
#     else:
#         img=Image.open(img_path1+'/'+file)
#         img.save(img_path+'/'+file)
#         # os.remove(img_path+'/'+file)
#         count+=1
# print(count)


#MNIST数据集颜色反转

# from PIL import Image
# import PIL.ImageOps
#
# #读入图片
#
#
# def trans_img(img_path,save_path):
#     for file in os.listdir(img_path):
#         if file=='.DS_Store':
#             os.remove(img_path+'/'+file)
#         else:
#
#             image = Image.open(img_path+'/'+file)
#             #反转颜色
#             inverted_image = PIL.ImageOps.invert(image)
#
#             #保存图片
#             inverted_image.save(save_path+'/m'+file)
#



# img_path='/Users/wywy/Desktop/MNIST'
# save_path='/Users/wywy/Desktop/MNIST1'
#
# trans_img(img_path,save_path)

# img_path='/Users/wywy/Desktop/xxx/3'
# save_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/train_cls/train_all'
#
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+file)

#
# img_path='/Users/wywy/Desktop/xxx'
# count=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file).convert('L')
#         img=img.resize((28,28),Image.ANTIALIAS)
#         img.save(img_path+'/'+str(count)+'.jpg')
#         count+=1
# print(count)



#删除增加的MNIST数据


# img_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/train_cls/train_all'
# count=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=list(file.split('.')[0].split('_')[0])
#         if 'm' in name:
#             os.remove(img_path+'/'+file)
#             count+=1
#
#
#
# print(count)


# img_path='/Users/wywy/Desktop/xxx/7'
# save_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/train_cls/train_all'
#
# count=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         img.save(save_path+'/'+file)
#         count+=1
# print(count)


# img_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/train_cls/train_all'
# count=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=list(file.split('.')[0])
#         # print(name)
#
#
#         if 'd' in name :
#             name2 = file.split('.')[0].split('_')[-1]
#             if name2=='7':
#                 os.remove(img_path+'/'+file)
#                 count+=1
# print(count)



#
#
# all_name=[]
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         name=file.split('.')[0].split('_')[-1]
#         if name=='1':
#             all_name.append(file)
# random.shuffle(all_name)
#
# print(len(all_name))
# cc=0
# for ii in all_name:
#     if cc<15000:
#         os.remove(img_path+'/'+ii)
#         cc+=1
# print(cc)

#删除错误数据

# img_path='/Users/wywy/Desktop/test_e'
# save_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/test_cls/test_all'
# count=0
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         os.remove(save_path+'/'+file)
#         count+=1
# print(count)



#增加识别错误数据
# save_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/train_cls/train_all'
# img_path='/Users/wywy/Desktop/train_e'
# cc=18624
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:
#         img=Image.open(img_path+'/'+file)
#         name=file.split('.')[0].split('_')[-1]
#         img.save(save_path+'/a'+str(cc)+'_'+name+'.jpg')
#         # print(save_path+'/a'+str(cc)+'_'+name+'.jpg')
#         cc+=1
#
# print(cc)


# img_path='/Users/wywy/Desktop/手写数字部分/截取正规数据部分/train_cls/train_all'
# save_path='/Users/wywy/Desktop/xxx/8'
# for file in os.listdir(save_path):
#     if file=='.DS_Store':
#         os.remove(save_path+'/'+file)
#     else:
#         img=Image.open(save_path+'/'+file)
#         img.save(img_path+'/'+file)
#         # name=file.split('.')[0].split('_')[-1]
#         # if name=='8':
#         #     os.remove(img_path+'/'+file)
#         #     # img=Image.open(img_path+'/'+file)
#         #     # img.save(save_path+'/'+file)

















from PIL import Image
import  os
import Augmentor
import cv2
import scipy.misc
import numpy as np

# img_path='/Users/wywy/Desktop/flase'
# save_path='/Users/wywy/Desktop/数据集/答题卡ID区域数据集/最后一次模型数据集（ID填涂部分）/new_train_cls'


#截取图片
def cut_img(img_path,save_path):
    for file in os.listdir(img_path):
        im = Image.open(img_path+'/'+file)
        region=im.crop((6,6,64-5,64-5))
        region.save(save_path+'/'+file)


#变换图片大小（resize）
def resize_picture(img_path):
    for i in os.listdir(img_path):
        if i=='.DS_Store':
            os.remove(img_path+'/'+i)
        else:
            im=Image.open(img_path+'/'+i)
            out=im.resize((64,318),Image.ANTIALIAS)
            out.save(img_path+'/'+i)
# resize_picture(img_path)

#图片分类：
def pictuer_clas(img_path):
    for file in os.listdir(img_path):
        if file == '.DS_Store':
            os.remove(img_path + '/' + file)
        else:
            name1=file.split('.')[0].split('_')[-1]
            if name1=='X':
                # im=Image.open(img_path+'/'+file)
                os.remove(img_path+'/'+file)
                # print(save_path+file)
                # im.save(save_path+file)

# pictuer_clas(img_path)
xx=[802,803,927,935,936,937,945,946,947,948,949,950,951,952,953,954,955,956,957]
test_id=xx[1]    #0-18
# num=9
save_path='/Users/wywy/Desktop/test_num_img'
img_path='/Users/wywy/Desktop/new_fill_img/{}'.format(test_id)
#更改文件路径
def change_path(img_path,save_path):
    count=13725
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            name=file.split('.')[0].split('_')[-1]
            # print(name)
            im=Image.open(img_path+'/'+file)
            im.save(save_path+'/'+str(count)+'_'+name+'.jpg')
            # print(save_path+'/'+str(count)+'_'+name+'.jpg')
            count+=1

    print(count)
            # print(save_path+'/'+file)
# change_path(img_path,save_path)

#在所有样本中找出原始数据
# def fin
# for file in os.listdir(save_path):
#     if file=='.DS_Store':
#         os.remove(save_path+'/'+file)
#     else:
#         name = file.split('.')[0].split('_')[0]
#         if int(name) >=30000:
#             os.remove(save_path+'/'+file)


#数据增强(扭曲旋转)
def picture_aug(img_path):
    p = Augmentor.Pipeline(img_path)
    p.random_distortion(probability=0.5, grid_width=1, grid_height=1, magnitude=1)
    p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
    p.sample(40000)


#数据增强（增加噪点，镜像）
def salt(img_path,n,save_path):
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            img=cv2.imread(img_path+'/'+file)
            for k in range(n):
                # 随机选择椒盐的坐标
                i = int(np.random.random() * img.shape[1])
                j = int(np.random.random() * img.shape[0])
                # 如果是灰度图
                if img.ndim == 2:
                    img[j, i] = 255
                    # 如果是RBG图片
                elif img.ndim == 3:
                    img[j, i, 0] = 255
                    img[j, i, 1] = 255
                    img[j, i, 2] = 255
            scipy.misc.imsave(save_path+'/'+'aug1_'+file, img)
# salt(img_path,1000,save_path)


#rename
def rename(img_path):
    xx=7239
    for file in os.listdir(img_path):
        if file == '.DS_Store':
            os.remove(img_path + '/' + file)
        else:
            im=Image.open(img_path+'/'+file)
            name1 = file.split('.')[0].split('_')[-1]
            im=Image.open(img_path+'/'+file)
            # # print('/Users/wywy/Desktop/flase_img/{}_{}.jpg'.format(xx,name1))
            im.save('/Users/wywy/Desktop/XX/{}_X.jpg'.format(xx))
            xx+=1
    print(xx)
# rename(img_path)



#删除图片
def remove_img(img_path):
    xx=0
    for file in os.listdir(img_path):
        name=file.split('.')[0].split('_')[-1]

        if name=='X':
            # print(img_path+'/'+file)
            os.remove(img_path+'/'+file)












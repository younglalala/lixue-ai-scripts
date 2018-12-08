import os

from PIL import  Image

up_img_path='..//num_up_img'
save_path='..//test_up_img'
# down_img_path='./num_down_img'
xx=0
for i in os.listdir(up_img_path):
    xx+=1
    im=Image.open(up_img_path+'/'+i)
    if xx<500:
        im.save(save_path+'/'+i)
        os.remove(up_img_path+'/'+i)
        print('完成保存{}    di1{}'.format(i,xx))


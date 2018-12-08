import Augmentor
import os
from PIL import  Image
import cv2


img_path='/Users/wywy/Desktop/训练集识别错误数据'
img_save_path='/Users/wywy/Desktop/训练完成模型/ID区域识别填涂部分/最后一次模型数据集（ID填涂部分）/train_cls'
def rename(img_path,save_path):
    flag=104524
    for file in os.listdir(img_path):
        if file=='.DS_Store':
            os.remove(img_path+'/'+file)
        else:
            name=file.split('.')[0].split('_')[1].split('e')[-1]
            img=Image.open(img_path+'/'+file)
            # print(save_path+'/'+str(flag)+'_'+name+'.jpg')
            img.save(save_path+'/'+str(flag)+'_'+name+'.jpg')
            flag+=1
    print(flag)






# rename(img_path,img_save_path)



















# for j in os.listdir(up_img_path):
#     if j=='.DS_Store':
#         os.remove(up_img_path+'/'+j)
#         print('xxx')
# xx=0
# for i in os.listdir(up_img_path):
#
#
#     xx+=1
#     # name=i.split('.')[0].split('_')[-1]
#     # print(up_img_path + '/output/' + i)
#     if xx<2000:
#         im = Image.open(up_img_path + '/' + i)
#         # print(im)
#         # if name=='9':
#         #     # print(save_path+str(0))
#         # print(save_path + str(0) + '/' + str(xx) + '_0.jpg')
#         # print(up_img_path+'/'+str(xx)+'_0.jpg')
#         # print(up_img_path+'/'+str(xx)+'_0.jpg')
#         # print(save_path+'/'+i)
#         # print()
#         im.save(save_path+'/'+i)
#         os.remove(up_img_path+'/'+i)
#         print(xx)

#
# p = Augmentor.Pipeline(up_img_path)
#
# p.rotate(probability=0.5, max_left_rotation=0.5, max_right_rotation=0.5)
# p.zoom(probability=0.1, min_factor=0.5, max_factor=1.0)
# p.sample(100)
# p.process()
# p.sample(100, multi_threaded=False)
# p = Augmentor.Pipeline("/path/to/images")
#
#0_original_24_0.jpg_45fe8e69-7bfa-47d7-87ed-45b0d7d0c742
# # Point to a directory containing ground truth data.
# # Images with the same file names will be added as ground truth data
# # and augmented in parallel to the original data.
# p.ground_truth("/path/to/ground_truth_images")
# # Add operations to the pipeline as normal:
# p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
# p.flip_left_right(probability=0.5)
# p.zoom_random(probability=0.5, percentage_area=0.7)
# p.flip_top_bottom(probability=0.5)
# p.sample(1000)


# p.random_distortion(probability=1, grid_width=3, grid_height=4, magnitude=4)
# p.flip_left_right(probability=0.2)
# p.flip_top_bottom(probability=0.2)
# p.sample(1000)


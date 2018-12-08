import Augmentor
import os
from PIL import  Image

# img_path='/Users/wywy/Desktop/train_img'   #7
# # save_img='/Users/wywy/Desktop/test1_img'
# xx=30000
# for i in os.listdir(img_path):
#     if i=='.DS_Store':
#         os.remove(img_path+'/'+i)
#     else:
#         im=Image.open(img_path+'/'+i)
#         out=im.resize((64,64),Image.ANTIALIAS)
#         out.save(img_path+'/'+i)
# #         name = i.split('.')[0].split('_')[-1]
# #         # print(name)
# #         im = Image.open(img_path + '/' + i)
# #         # if name=='9':
# #         # print(save_img+'/'+str(xx)+'_'+name+'.jpg')
# #         # print(save_img+'/'+str(xx)+'_'+name+'.jpg')
# #         im.save(save_img+'/'+str(xx)+'_'+name+'.jpg')
# #         xx+=1
# # #             # os.remove(img_path+'/'+i)
# #
# #     # print('/Users/wywy/Desktop/train_img/'+str(xx)+'_'+name+'.jpg')
# #     # xx+=1
# # print(xx)



# test_img='..//test_img'
# save_path='/Users/wywy/Desktop/test_img'
    # name=i.split('.')[0].split('_')[-1]
    # print(up_img_path + '/output/' + i)
    # if xx<2000:
    #     im = Image.open(up_img_path + '/' + i)
    #     # print(im)
    #     # if name=='9':
    #     #     # print(save_path+str(0))
    #     # print(save_path + str(0) + '/' + str(xx) + '_0.jpg')
    #     # print(up_img_path+'/'+str(xx)+'_0.jpg')
    #     # print(up_img_path+'/'+str(xx)+'_0.jpg')
    #     # print(save_path+'/'+i)
    #     # print()
    #     im.save(save_path+'/'+i)
    #     os.remove(up_img_path+'/'+i)
    #     print(xx)
# for f in os.listdir(img_path):
#     name=f.split('.')[0].split('_')[-1]

img_path='/Users/wywy/Desktop/all_choice'
# save_path='/Users/wywy/Desktop/ok_train_num/train_num_aug3'
#
# for file in os.listdir(img_path):
#     if file=='.DS_Store':
#         os.remove(img_path+'/'+file)
#     else:


p = Augmentor.Pipeline(img_path)

p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
        # p.zoom(probability=0.1, min_factor=0.5, max_factor=1.0)
        # p.sample(100)
        # p.process()
        # p.sample(100, multi_threaded=False)
        # p = Augmentor.Pipeline("/path/to/images")
        #
        # Point to a directory containing ground truth data.
        # Images with the same file names will be added as ground truth data
        # and augmented in parallel to the original data.
        # p.ground_truth("/path/to/ground_truth_images")
        # Add operations to the pipeline as normal:
        # p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
        # p.flip_left_right(probability=0.5)
        # p.zoom_random(probability=0.5, percentage_area=0.7)
        # p.flip_top_bottom(probability=0.5)
        # p.sample(1000)


p.random_distortion(probability=1, grid_width=2, grid_height=2, magnitude=2)
p.flip_left_right(probability=0.2)
p.flip_top_bottom(probability=0.2)
p.sample(2000)


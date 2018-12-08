import numpy as np
import cv2
import os
from tornado.options import options
img_path='/Users/wywy/Desktop/save'
for i in os.listdir(img_path):
    if i == '.DS_Store':
        pass
    else:

        aa=cv2.imread(img_path+'/'+i)
        h,w,c=np.shape(aa)
        lable_x=30.
        lable_y=14.
        for j in range(9):
            lable_y+=31
        # print(lable_center_y)
            print(int(lable_x),int(lable_y))
# def parse_column(self, img):
#     """ 解析每一列
#
#     :param img: 每一列的图片
#     :return: 解析后的值
#     """
#     # 切除黑框边
#     img = img[2:-2, 2:-2]
#
#     # 灰度图
#     img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 二值化则灰度在大于50的像素其值将设置为255，其它像素设置为0。
#
#     img_grey[:, :2] = np.ones((img.shape[0], 2), dtype=np.uint8) * 255
#     img_grey[:, -2:] = np.ones((img.shape[0], 2), dtype=np.uint8) * 255
#
#     width_erode = int(img.shape[1] * 0.15)
#
#     threshold = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[
#         1]  # THRESH_BINARY_INV黑白二值翻转。cv2.THRESH_OTSU直方图的二值化
#
#     img_grey = cv2.dilate(threshold, np.ones((3, 3), dtype=np.uint8))
#
#     img_grey = cv2.erode(img_grey, np.ones((5, width_erode), dtype=np.uint8))
#
#     _, contours, _ = cv2.findContours(img_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     rects = []
#     for contour in contours:
#         rect = cv2.boundingRect(contour)  # 边缘检测，返回的是矩形框的想，x,y,w,h
#         x, y, w, h = rect
#         if not (w > img.shape[1] * 0.3 and img.shape[1] * 0.1 < h < img.shape[1]):
#             continue
#         rects.append(rect)
#
#     # 将坐标从上往下排序
#     # sorted(rects, key=operator.itemgetter(1))
#
#     rected = None
#
#     for rect in rects:
#         x, y, w, h = rect
#         # mask有rect这个区域
#         mask = np.zeros(threshold.shape, dtype=np.uint8)
#         cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
#
#         # mask与二值化后的图片
#         mask = cv2.bitwise_and(threshold, threshold, mask=mask)
#
#         total = cv2.countNonZero(mask)
#         # debug_show("mask {}".format(total), mask)
#
#         if rected is None or total > rected[0]:
#             rected = (total, rect)
#
#     if rected is None or rected[0] < img.shape[1] * img.shape[1] * 0.25 * 0.25:
#         return None, None
#
#     height, width = img.shape[:2]
#     x, y, w, h = rected[1]
#
#     idx = int((y + h / 2.0) / (height / 10.0))
#     if not 0 <= idx <= 9:
#         return None, None
#
#     img_copy = img.copy()
#     cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     # debug_show("column number {}".format(idx), img_copy)
#
#     return idx, rected[1], img

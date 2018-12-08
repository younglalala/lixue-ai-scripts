# coding=utf-8
import operator

import cv2
import numpy as np
from tornado.options import options

import settings
from .image_utils import debug_show


class IdParser(object):
    ID_NUM = 5

    def __init__(self, img):
        self.img = img
        self.height, self.width = self.img.shape[:2]

    @property
    def id_height(self):
        return 0.73

    def parse(self):
        rects = self.find_column()
        if len(rects) != self.ID_NUM:
            raise NotImplementedError()

        rects = sorted(rects)  # 按x坐标排序

        result = ""
        num_rect_list = []
        for rect in rects:
            x, y, w, h = rect
            img = self.img[y: y + h, x: x + w]
            num, num_rect = self.parse_column(img)

            if num is None:
                result = result + "X"
                num_rect_list.append((0, 0, 0, 0))
            else:
                result = result + str(num)
                x1, y1, w1, h1 = num_rect
                num_rect_list.append((x + x1, y + y1, w1, h1))

        return result, num_rect_list

    def find_column(self):
        """查找每一列

        :return: 每一列的rect坐标
        """
        rects = []

        for index in range(self.ID_NUM):
            x = int(1.0 * index * self.img.shape[1] / self.ID_NUM)
            y = int((1 - self.id_height) * self.img.shape[0])
            w = int(1.0 * self.img.shape[1] / self.ID_NUM)
            h = int(self.id_height * self.img.shape[0])

            rects.append((x, y, w, h))

        return rects

    def parse_column(self, img):
        """ 解析每一列

        :param img: 每一列的图片
        :return: 解析后的值
        """
        # 切除黑框边
        img = img[2:-2, 2:-2]

        # 灰度图
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_grey[:, :2] = np.ones((img.shape[0], 2), dtype=np.uint8) * 255
        img_grey[:, -2:] = np.ones((img.shape[0], 2), dtype=np.uint8) * 255

        width_erode = int(img.shape[1] * 0.15)

        threshold = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        img_grey = cv2.dilate(threshold, np.ones((3, 3), dtype=np.uint8))
        img_grey = cv2.erode(img_grey, np.ones((5, width_erode), dtype=np.uint8))

        _, contours, _ = cv2.findContours(img_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for contour in contours:
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            if not (w > img.shape[1] * 0.3 and img.shape[1] * 0.1 < h < img.shape[1]):
                continue
            rects.append(rect)

        # 将坐标从上往下排序
        sorted(rects, key=operator.itemgetter(1))

        rected = None

        for rect in rects:
            x, y, w, h = rect
            # mask有rect这个区域
            mask = np.zeros(threshold.shape, dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

            # mask与二值化后的图片
            mask = cv2.bitwise_and(threshold, threshold, mask=mask)

            total = cv2.countNonZero(mask)
            debug_show("mask {}".format(total), mask)

            if rected is None or total > rected[0]:
                rected = (total, rect)

        if rected is None or rected[0] < img.shape[1] * img.shape[1] * 0.25 * 0.25:
            return None, None

        height, width = img.shape[:2]
        x, y, w, h = rected[1]

        idx = int((y + h / 2.0) / (height / 10.0))
        if not 0 <= idx <=9:
            return None, None

        img_copy = img.copy()
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        debug_show("column number {}".format(idx), img_copy)

        return idx, rected[1]

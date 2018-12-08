# coding=utf-8
import os
import logging
import operator

import cv2
import numpy as np
from tornado.options import options

import settings


def debug_show(winname, mat, level="DEBUG"):
    if logging.getLevelName(level.upper()) < logging.getLevelName(options.logging.upper()):
        return

    height, width = mat.shape[:2]
    if height > 1200:
        mat = cv2.resize(mat, (int(width * 1200 / height), 1200))

    logging.info("%s size %s, %s", winname, mat.shape[0], mat.shape[1])

    cv2.imshow(winname, mat)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)


def save_image(file_path, img):
    """保存图片

    :param file_path: 路径
    :param img: 图片
    :return: 保存后的路径
    """
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    cv2.imwrite(file_path, img)
    return file_path


def find_black_symbol(img_grey, width_erode=10, height_erode=10):
    """寻找黑色标记点

    :param img: 原图
    :return: 灰度图，只有黑色标记点
    """
    # 二值化
    img_grey = cv2.threshold(img_grey, 0, 256, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    debug_show("Image Threshold", img_grey)

    # Dilation 膨胀
    img_grey = cv2.dilate(img_grey, kernel=np.ones((int(height_erode * 0.2), int(width_erode * 0.2)), dtype=np.uint8))
    debug_show("Image Dilate", img_grey)

    # Erode 腐蚀
    img_grey = cv2.erode(
        img_grey,
        kernel=np.ones((int(height_erode * 1.2), int(width_erode * 1.2)), dtype=np.uint8),
        iterations=1
    )
    debug_show("Image Erode", img_grey)

    # Dilation 膨胀
    img_grey = cv2.dilate(img_grey, kernel=np.ones((height_erode, width_erode), dtype=np.uint8))
    debug_show("Image Dilate", img_grey)

    return img_grey


def center_of_gravity(contour):
    """计算重心

    :param contour: 轮廓
    :return: 重心坐标
    """

    M = cv2.moments(contour)
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def gamma_trans(img_grey):
    """gamma变换

    :param img_grey: 原图，灰度图
    :return: 新的图片
    """
    # 求平均亮度
    brightness = cv2.mean(img_grey)[0]
    # TODO 这段代码可以优化
    # gamma变换，把亮度低的地方变得更低，所以取100而不是128。128是256的一半，可能是平均亮度
    gamma = 1 + (brightness - 100) / 100
    if gamma > 1.5:
        gamma = gamma + 1

    # gamma 变换
    img_grey = np.float32(img_grey)
    img_grey = cv2.pow(img_grey, gamma)
    cv2.normalize(img_grey, img_grey, 0, 255, cv2.NORM_MINMAX)
    img_grey = cv2.convertScaleAbs(img_grey)

    return img_grey


def equalize_hist(img_grey):
    """直方图均衡化

    :param img_grey: 原图，灰度图
    :return: 新的图片
    """
    hist = cv2.calcHist(
        [img_grey],  # 计算图像的直方图
        [0],  # 使用的通道
        None,  # 没有使用mask
        [256],  # it is a 1D histogram
        [0.0, 255.0],
    )

    min_bin_no, max_bin_no = 0, 255

    for bin_no, bin_value in enumerate(hist):
        if bin_value != 0:
            min_bin_no = bin_no
            break

    for bin_no, bin_value in reversed(list(enumerate(hist))):
        if bin_value != 0:
            max_bin_no = bin_no
            break

    if min_bin_no == max_bin_no:
        return img_grey

    # 生成查找表，参考文献 《Opencv2 Computer Vision Application Programming Cookbook》 第四章第2节
    lut = np.zeros(256, dtype=img_grey.dtype)
    for index, value in enumerate(lut):
        if index < min_bin_no:
            lut[index] = 0
        elif index > max_bin_no:
            lut[index] = 255
        else:
            lut[index] = int(255.0 * (index - min_bin_no) / (max_bin_no - min_bin_no) + 0.5)

    return cv2.LUT(img_grey, lut)


def sorted_corner_points(points):
    """四个角，坐标排序

    :param points:
    :return:
    """
    points = sorted(points)
    left, right = points[:2], points[2:]

    left = sorted(left, key=operator.itemgetter(1))
    right = sorted(right, key=operator.itemgetter(1))

    left_top, left_bottom = left
    right_top, right_bottom = right

    return [left_top, right_top, left_bottom, right_bottom]


def warpPerspective(img, M):
    """透视变换

    :param img: 原图
    :param M: 变换矩阵
    :return: 返回图片
    """
    height, width = img.shape[:2]
    if len(img.shape) == 2:
        border_value = 255
    else:
        border_value = (255, 255, 255)

    return cv2.warpPerspective(img, M, (width, height), borderValue=border_value)


def scale_points(points, scale):
    """对一系列点坐标进行缩放

    :param points:
    :param scale:
    :return:
    """
    return [(x * scale, y * scale) for x, y in points]


def points_add(p1, p2):
    """两点坐标相加

    :param p1: 点1
    :param p2: 点2
    :return:
    """
    return tuple(map(sum, zip(p1, p2)))


def sorted_rect(rect_list, style="column"):
    """排列矩形

    :param rect_list: 矩形列表，矩形的格式（x, y, w, h）
    :param style: column or row, 按列或者按行排列
    :param delta: 超过delta，就算下一行或者下一列
    :return: 排序过的矩形矩阵
    """

    if not rect_list:  # 为空
        return rect_list

    if style == "row":  # 如果是按行排列，跟按列排列相反，坐标交换下即可
        rect_list = [(y, x, h, w) for x, y, w, h in rect_list]

    rect_list = sorted(rect_list)

    # 当前列最左边的x坐标
    current_column = [rect_list[0]]
    rect_matrix = [current_column]

    for rect in rect_list[1:]:
        if rect[0] - current_column[0][0] >= current_column[0][2]:  # 已经换列
            current_column = [rect]
            rect_matrix.append(current_column)

        else:  # 没有换列
            current_column.append(rect)

    # 按第二个维度来排序
    rect_matrix = [sorted(column, key=operator.itemgetter(1)) for column in rect_matrix]

    if style == "row":  # 如果是按行排列，跟按列排列相反，坐标交换下即可
        tmp = []
        for column in rect_matrix:
            new_column = [(x, y, w, h) for y, x, h, w in column]
            tmp.append(new_column)
        rect_matrix = tmp

    rect_result = []
    for rect_column in rect_matrix:
        rect_result.extend(rect_column)

    return rect_result

import logging
import math
import operator

import cv2
import numpy as np

import settings
from melon.image_utils import debug_show


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def find_squares(img, is_not=False):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        if is_not:
            gray = cv2.bitwise_not(gray)
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)

            # debug_show("bin {}".format(thrs), bin, level="WARN")
            bin, contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and 3000 > cv2.contourArea(cnt) > 1500 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4]) for i in range(4)])
                    max_cos_45 = np.max([angle_cos(cnt[(i+1) % 4], cnt[(i+2) % 4], cnt[i]) for i in range(4)])
                    if max_cos < 0.2 and math.fabs(max_cos_45 - 0.70710678118) < 0.1:
                        squares.append(cnt)
    return squares


def get_centroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    return cx, cy


def find_father(labels, idx):
    if labels[idx] == -1:
        return idx

    father = find_father(labels, labels[idx])
    labels[idx] = father

    return father


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


def main():
    img = cv2.imread(r"C:\Users\37239\Desktop\tmp\mobile\24.jpg")
    # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # debug_show("img_grey grey", img_grey, level="WARN")

    # img_grey = cv2.threshold(img_grey, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
    # debug_show("img_grey grey", img_grey, level="WARN")

    # img_grey = cv2.morphologyEx(img_grey, cv2.MORPH_OPEN, np.ones((50, 50), dtype=np.int32))
    # debug_show("img_grey grey", img_grey, level="WARN")

    squares = find_squares(img)
    squares += find_squares(img, is_not=True)

    img_copy = img.copy()
    img_copy = cv2.drawContours(img_copy, squares, -1, (0, 255, 0), 3)
    debug_show("img_copy", img_copy, level="WARN")

    centroids = []

    img_bin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.threshold(img_bin, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
    black_squares = []
    for square in squares:
        x, y, w, h = cv2.boundingRect(square)
        if cv2.countNonZero(img_bin[y: y + h, x: x + w]) > 200:
            black_squares.append(square)
    squares = black_squares

    for square in squares:
        centroids.append(get_centroid(square))

    centroid_labels = [-1] * len(centroids)
    for idx1, centroid_1 in enumerate(centroids):
        for idx2, centroid_2 in enumerate(centroids):
            if idx1 >= idx2:
                continue

            x1, y1 = centroid_1
            x2, y2 = centroid_2
            if (y2 - y1) ** 2 + (x2 - x1) ** 2 < 100:
                centroid_labels[idx2] = find_father(centroid_labels, idx1)
            else:
                pass
                # logging.warning("%s != %s", centroid_1, centroid_2)

    centroid_group = {}
    for idx, centroid in enumerate(centroids):
        father = find_father(centroid_labels, idx)
        if father not in centroid_group:
            centroid_group[father] = []
        centroid_group[father].append(centroid)

    logging.info("%s", len(centroid_group))


    points = []
    for centroid_list in centroid_group.values():
        logging.warning("centroid_list %s", centroid_list)
        x, y = zip(*centroid_list)
        logging.warning("x %s y %s", x, y)
        x = int(sum(x) / len(x))
        y = int(sum(y) / len(y))
        points.append((x, y))

    points = sorted(points)
    points = points[:2] + points[-2:]

    points = sorted_corner_points(points)

    height, width = 3508, 2480
    x, y, w, h = (
        (88.0 + 59.0 / 2) / 2480.0,
        (88.0 + 59.0 / 2) / 3508.0,
        (2480.0 - 88.0 * 2 - 59.0) / 2480.0,
        (3508.0 - 88.0 * 2 - 59.0) / 3508.0,
    )
    x, y, w, h = x * width, y * height, w * width, h * height
    correction_points = (x, y), (x + w, y), (x, y + h), (x + w, y + h)

    M = cv2.getPerspectiveTransform(np.float32(points), np.float32(correction_points))

    border_value = (255, 255, 255)

    img_warped = cv2.warpPerspective(img, M, (width, height), borderValue=border_value)
    debug_show("warp", img_warped, level="WARN")

    cv2.imwrite(r"C:\Users\37239\Desktop\tmp\mobile\xxx.jpg", img_warped)

    x, y, w, h = 14 / 210.0, 95 / 297.0, 180 / 210.0, 69 / 297.0
    x, y, w, h = int(x * 2480), int(y * 3508), int(w * 2480), int(h * 3508)
    box_block_img = img_warped[y: y + h, x: x + w]

    debug_show("box block", box_block_img, level="WARN")

    img_grey = cv2.cvtColor(box_block_img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 10))
    img_grey = cv2.morphologyEx(img_grey, cv2.MORPH_OPEN, kernel=kernel)
    img_grey = cv2.morphologyEx(img_grey, cv2.MORPH_CLOSE, kernel=kernel)
    debug_show("box block", img_grey, level="WARN")
    img_grey = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    debug_show("box block", img_grey, level="WARN")

    edges = cv2.Canny(img_grey, 20, 80)
    linesP = cv2.HoughLinesP(edges, 1, np.pi/45, 45, 20, 20)

    if linesP is not None:
        img_copy = box_block_img.copy()
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(img_copy, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
        debug_show("box block with line", img_copy, level="WARN")



if __name__ == "__main__":
    main()

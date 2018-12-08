import logging
import urllib
import scipy.misc
import matplotlib.pyplot as plt
from num_cut import *

import requests
import cv2
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tornado.options import define, options
import  os

from melon.id_parser import IdParser
ex_id=1000
define("exam_id", ex_id, "exam_id")

import settings


ENGINE = create_engine(settings.DB['conn_param'], echo=settings.DB['debug'])


DBSession = sessionmaker(bind=ENGINE)
save_path='/Users/wywy/Desktop/id_rename/'

def main():
    session = DBSession()

    user_dict = {}
    result = session.execute("""
        SELECT
            id, username
        FROM
            sys_user
        WHERE
            deleted_at = -1
    """)
    for item in result:
        user_dict[item['id']] = item['username']

    result = session.execute(f"""
        SELECT
            id, layout
        FROM
            qb_answer_sheet_structure 
        WHERE
            exam_id = {options.exam_id}
        AND deleted_at = -1
    """)
    result = next(result)
    layout = int(result["layout"])
    answer_sheet_structure_id = int(result["id"])

    result = session.execute(f"""
        SELECT
            x, y, w, h
        FROM
            qb_answer_sheet_structure_content
        WHERE
            answer_sheet_structure_id = {answer_sheet_structure_id}
        AND type = 'registerationNumber'
        AND deleted_at = -1
    """)
    result = next(result)
    x, y, w, h = result["x"], result["y"], result["w"], result["h"]

    result = session.execute(f"""
        SELECT
            id, user_id, front_original_file_path
        FROM
            ex_marking_img_file
        WHERE
            deleted_at = -1
        AND exam_id = {options.exam_id}
    """)

    if layout == 1:
        paper_width, paper_height = 210, 297
    else:
        paper_width, paper_height = 420, 297
    flag = 0
    for image_item in result:
        part_key = image_item['front_original_file_path']

        r = requests.get(f"http://static.lixueweb.com/{part_key}?x-oss-process=image/info")
        image_info = r.json()
        image_width = int(image_info["ImageWidth"]["value"])
        image_height = int(image_info["ImageHeight"]["value"])

        cx = int(x * image_width / paper_width)+8
        cy = int(y * image_height / paper_height)+10
        cw = int(w * image_width / paper_width)
        ch = int(h * image_height / paper_height)

        oss_process_string = f"?x-oss-process=image/crop,x_{cx},y_{cy},w_{cw},h_{ch}"
        image_url = f"http://static.lixueweb.com/{part_key}{oss_process_string}"
        print('xxx')
        file_path = "/Users/wywy/Desktop/id_img/{}.jpg".format(image_item['id'])
        # logging.info("image_url: %s file_path %s", image_url, file_path)

        response = requests.get(image_url)
        with open(file_path, "wb") as f:
            f.write(response.content)

        img = cv2.imread(file_path)


        id_parser = IdParser(img)
        number_parser=NumParser(img)
        num_img=number_parser.get_img()

        result, num_rect_list = id_parser.parse()    #result，数字，num_rect_list，坐标,num_img返回截取的图片信息

        username = user_dict[image_item["user_id"]]
        register_number = username[-5:]
        lable_number=list(result)
        lable2_number=list(register_number)

        all_img='/Users/wywy/Desktop/all_img/{}/{}.jpg'.format(ex_id,register_number)
        scipy.misc.imsave(all_img, img)
        flag += 1
        for k in range(5):
            save_num_img=num_img[k]
            # save_img=id_img[k]
            img_lable=lable_number[k]
            img_lable1=lable2_number[k]
            # path_id = "/Users/wywy/Desktop/num_down_img/{}_{}.jpg".format(flag,img_lable)
            path_num = "/Users/wywy/Desktop/num_up_img/{}_{}.jpg".format(flag, img_lable1)
            # print(path_num)
            #
            # scipy.misc.imsave(path_id, save_img)
            scipy.misc.imsave(path_num,save_num_img)


        if result != register_number:
            logging.warning("parser %s real %s", result, register_number)
        else:
            logging.info("label %s", result)


if __name__ == "__main__":
    main()

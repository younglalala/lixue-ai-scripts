import logging
import urllib

import cv2
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tornado.escape import utf8, to_unicode
from tornado.options import define, options

define("exam_id", 799, "exam_id")

import settings


ENGINE = create_engine(settings.DB['conn_param'], echo=settings.DB['debug'])


DBSession = sessionmaker(bind=ENGINE)


class FacePlusPlusDemo(object):
    def __init__(self, image_url):
        self.image_url = image_url

        self.result = []

    def process(self):
        r = requests.post(
            "https://api-cn.faceplusplus.com/imagepp/v1/recognizetext",
            data=dict(
                api_key="MqA3O6Xf3vZnvoeA1klln2rHPsoleO6i",
                api_secret="ZX0bKtuijMgFacbPvvXltSXWyl_g-ENK",
                image_url=self.image_url,
            )
        )

        if len(r.text) < 100:
            logging.warning("text: %s", r.text)

        for child_object in r.json()['result']:
            self.process_child_object(child_object)

    def process_child_object(self, child_object):
        if "position" in child_object and "value" in child_object and not child_object["child-objects"]:
            pts = [(pt['x'], pt['y']) for pt in child_object["position"]]
            self.result.append((pts, child_object['value']))

        for child_object in child_object['child-objects']:
            self.process_child_object(child_object)



def main():
    session = DBSession()
    result = session.execute(f"""
        SELECT   
	      ex_marking_part_img_file.part_file_path AS part_file_path
        FROM
          ex_marking_part_img_file
        LEFT JOIN ex_marking_img_file ON ex_marking_part_img_file.marking_img_id = ex_marking_img_file.id 
        WHERE
          ex_marking_img_file.exam_id = {options.exam_id}
        AND ex_marking_img_file.deleted_at = -1
        AND ex_marking_part_img_file.deleted_at = -1
    """)

    for item in result:
        part_key = item['part_file_path']
        image_url = f"http://static.lixueweb.com/{part_key}?x-oss-process=image/resize,w_800,h_800,limit_1"

        logging.info("image_url: %s", image_url)

        demo = FacePlusPlusDemo(image_url)
        try:
            demo.process()
        except:
            continue

        tmp_filename = "tmp/{}".format(part_key.replace("/", "_"))

        # download file
        urllib.request.urlretrieve(image_url, tmp_filename)

        # draw things
        img = Image.open(tmp_filename)
        for pts, value in demo.result:
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("simsun.ttc", 16)
            draw.text((pts[0][0], pts[0][1] - 16), value, (255, 0, 0), font=font)

        img.save(tmp_filename.replace(".jpg", "_draw.jpg"))


if __name__ == "__main__":
    main()
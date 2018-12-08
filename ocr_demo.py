
# from aip import AipOcr
#
# """ 你的 APPID AK SK """
# APP_ID = '11571640'
# API_KEY = 'N374Y5CqfQ2WWHSROYFo8HoS'
# SECRET_KEY = 'T5yj6hDrvSVvTHFhoZ1Iu9OrdF6ZxcGf'
#
# client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
#
# def get_file_content(filePath):
#     with open(filePath, 'rb') as fp:
#         return fp.read()
#
# image = get_file_content('/Users/wywy/Desktop/id_number_1.jpg')
#
# """ 调用通用文字识别, 图片参数为本地图片 """
# client.basicGeneral(image)
#
# """ 如果有可选参数 """
# options = {}
# options["language_type"] = "CHN_ENG"
# options["detect_direction"] = "true"
# options["detect_language"] = "true"
# options["probability"] = "true"
#
# """ 带参数调用通用文字识别, 图片参数为本地图片 """
# aa=client.basicGeneral(image, options)
# print(aa)


#!/usr/bin/python
# -*- coding: UTF-8 -*-
import urllib.request
from urllib import parse

import time
import urllib
import json
import hashlib
import base64


def main():
    f = open("/Users/wywy/Desktop/id_number_1.jpg", 'rb')
    file_content = f.read()
    base64_image = base64.b64encode(file_content)
    body = parse.urlencode({'image': base64_image})

    url = 'http://webapi.xfyun.cn/v1/service/v1/ocr/handwriting'
    api_key = '1add28aca4ea6ce0c8c346c44f93f0e5'
    param = {"language": "en", "location": "true"}

    x_appid = '5b30ad56'
    x_param = base64.b64encode(json.dumps(param).replace(' ', ''))
    x_time = int(int(round(time.time() * 1000)) / 1000)
    x_checksum = hashlib.md5(api_key + str(x_time) + x_param).hexdigest()
    x_header = {'X-Appid': x_appid,
                'X-CurTime': x_time,
                'X-Param': x_param,
                'X-CheckSum': x_checksum}
    req = urllib.request.Request(url, body, x_header)
    result = urllib.request.urlopen(req)
    result = result.read()
    print(result)
    return


print(main())










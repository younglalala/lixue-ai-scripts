import json
import urllib.request
import base64

import requests


host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=FqN1wpE2YpYpS3aRwBXigbZN&client_secret=oHug1dblHf4SXTYHcVngfZtZlV3CQvCH'
request = urllib.request.Request(host)
request.add_header('Content-Type', 'application/json; charset=UTF-8')
response = urllib.request.urlopen(request)
content = response.read()
if (content):
    content = json.loads(content)

access_token = content["access_token"]

image_body = base64.b64encode(open("/Users/wywy/Desktop/test_resize/7_68_165_463.jpg", "rb").read()).decode('ascii')

r = requests.post(
    # "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic",
    # "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic",
    "https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting",
    headers={"Content-Type": "application/x-www-form-urlencoded"},
    data=dict(
        access_token=access_token,
        image=image_body,
    )
)

print(r.text)

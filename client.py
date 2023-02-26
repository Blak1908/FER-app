# importing the requests library
import requests
import base64
import io
import json
import cv2
from PIL import Image
# api-endpoint http://127.0.0.1:8000/api/v1/fer/docs#/fer/predict_api_v1_fer_predict_post
URL = "http://127.0.0.1:8000/api/v1/fer/predict"
  
import requests

with open("PrivateTest_8002268.jpg", "rb") as image2string:
    converted_bytes = base64.b64encode(image2string.read())
    
img_str = converted_bytes.decode("utf-8") 
data_json = {"data": img_str}
print(data_json)
r = requests.get(URL, params=data_json).json
print(r)

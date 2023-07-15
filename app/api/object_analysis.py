from fastapi import FastAPI
from pydantic import BaseModel
import os, io
from app.core.settings import get_settings
import cv2
import base64
import numpy as np
from PIL import Image
from typing import List

settings = get_settings()

app = FastAPI()

path_temp = settings.TEMP_PATH
src_folder_name = settings.SCR_FOLDER_NAME
src_folder_path = f'{path_temp}/{src_folder_name}'

if not os.path.exists(src_folder_path):
    os.makedirs(src_folder_path)

class Item(BaseModel):
    images: List[str]

@app.post("/api/v1/object-analysis/")
async def create_item(item: Item):
    import pdb; pdb.set_trace()
    status = 200
    try:
        for i, image_data in enumerate(item.images):
            decoded_image = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(decoded_image))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{src_folder_path}/{i}.jpg", image)
    except Exception as e:
        print(e)
        status = 400
        
    return status
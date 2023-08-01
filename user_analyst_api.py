from fastapi import FastAPI
from pydantic import BaseModel
import os, io
from app.core.settings import get_settings
import cv2
import base64
import numpy as np
from PIL import Image
from typing import List

from app.modules.user_analyst.user_analyst_processing import deepface_processing

settings = get_settings()

app = FastAPI()

path_temp = settings.TEMP_PATH
src_folder_name = settings.SCR_FOLDER_NAME
database_image_path = settings.DATABASE_IMAGE_PATH
src_folder_path = f'{path_temp}/{src_folder_name}'

if not os.path.exists(src_folder_path):
    os.makedirs(src_folder_path)

class Item(BaseModel):
    images: List[str]
    isRequire_analys: bool

class Information(BaseModel):
    images: List[str]
    name: str

@app.post("/api/v1/object-analysis/")
def create_item(item: Item):
    status = 200
    print(item.isRequire_analys)
    images = []
    try:
        for i, image_data in enumerate(item.images):
            decoded_image = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(decoded_image))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            images.append(image)
            cv2.imwrite(f"{src_folder_path}/{i}.jpg", image)
         
        result = deepface_processing(flag=True, src_folder_path=src_folder_path)
            
    except Exception as e:
        print(e)
        status = 400
        
    return {"result": result, "status": status}

@app.post("/api/v1/create-user/")
def create_user(information: Information):
    status = 200
    images = []
    
    name = information.name
    db_user_path = f"{database_image_path}/{name}" 
    
    if not os.path.exists(db_user_path):
        print(f"Create DB for User {name}: {db_user_path}")
        os.makedirs(db_user_path)

    try:
        for i, image_data in enumerate(information.images):
            decoded_image = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(decoded_image))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            images.append(image)
            cv2.imwrite(f"{db_user_path}/{name}-{i}.jpg", image)
         
        result = deepface_processing(name)
            
    except Exception as e:
        print(e)
        status = 400
        
    return {"result": result, "status": status}

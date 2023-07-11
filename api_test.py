from fastapi import FastAPI
from pydantic import BaseModel
import cv2
from starlette.responses import StreamingResponse
import io
import os 
from app.core.utils import download_folder_model
from app.core.settings import get_settings

settings = get_settings()

app = FastAPI()

path_temp = settings.TEMP_PATH
src_folder_name = settings.SCR_FOLDER_NAME
src_folder_path = f'{path_temp}/{src_folder_name}'

if not os.path.exists(src_folder_path):
    os.makedirs(src_folder_path)



class Item(BaseModel):
    driverUrl: str

@app.get("/api/v1/object-analysis/")
async def root(item: Item):
    print("Driver Url: ", item.driverUrl)
    status = download_folder_model(item.driverUrl,src_folder_path)
    return {"status": status, "driver url": item.driverUrl}


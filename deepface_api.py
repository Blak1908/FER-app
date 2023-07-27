from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.modules.user_recognition.deepface_transform import deepface_processing
from typing import List

app = FastAPI()

@app.post("/api/v1/ai-chatbot-comsumer")
async def message_consumer(files: List[UploadFile] = File(...)):
    result = deepface_processing(db_path="/home/cuongacpe/Documents/AI-Chatbot-Synthesis/app/modules/user_recognition/user/database",
                                 models="Facenet", backends='dlib', distances='euclidean_l2',flag=True)
    return result
    
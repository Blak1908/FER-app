from fastapi import FastAPI
from pydantic import BaseModel
import os
from app.core.settings import get_settings
import base64
import numpy as np
from scipy.io import wavfile
from app.modules.speech2text.s2t_transformer import s2t_processing

settings = get_settings()

app = FastAPI()

path_temp = settings.TEMP_PATH
src_folder_name = settings.SCR_FOLDER_NAME
src_folder_path = f'{path_temp}/{src_folder_name}'

fs = 48000

if not os.path.exists(src_folder_path):
    os.makedirs(src_folder_path)
class audioAnalys(BaseModel):
    audio: str

@app.post("/api/v1/speech2text")
async def speech2text(audioAnalys: audioAnalys):
    status = 200
    audio_bytes = base64.b64decode(audioAnalys.audio)
    # Convert bytes back to NumPy array
    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
    scaled_audio = np.int16(audio_array * 32767)
    tmp_audio_file = f"{path_temp}/output_audio.wav"
    wavfile.write(tmp_audio_file, fs, scaled_audio)
    result = s2t_processing(tmp_audio_file)
    print("result: ", result['text'])
    # Encode the string to bytes (UTF-8 encoding)
    encoded_bytes = result['text'].encode('utf-8')
    return {"status": status, "text": encoded_bytes}
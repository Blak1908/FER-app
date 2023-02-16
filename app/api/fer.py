import os 
import pathlib
from fastapi import FastAPI
from fastapi import FastAPI
import io
import sys
cwd = os.getcwd()

def get_root_project(cwd):
    user_path = ''
    path_list = cwd.split('/')
    for i in path_list:
        if str(i) == 'app':
            return user_path
        user_path = str(user_path) +  str(i) + '/'
    return False

user_path = get_root_project(cwd)


sys.path.append(user_path)

from app.core.settings import get_setting
from app.core.fer_predict import predict


app = FastAPI()


BASE_DIR = pathlib.Path(__file__).resolve().parent



@app.get('/')
def first_api_test():
    return {'hello': 'world'}


@app.post('/predict')
def predict(img):
    emotion = predict(img)
    return {'Result predict: ': emotion}
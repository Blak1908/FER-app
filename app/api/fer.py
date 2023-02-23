import pathlib
from fastapi import FastAPI

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
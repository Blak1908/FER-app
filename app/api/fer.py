import pathlib
from fastapi import APIRouter


from app.core.fer_predict import predict


router = APIRouter()


BASE_DIR = pathlib.Path(__file__).resolve().parent





@router.post('/predict')
def predict(img):
    emotion = predict(img)
    return {'Result predict: ': emotion}
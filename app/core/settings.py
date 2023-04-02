import os
from pydantic import BaseSettings
from starlette.config import Config





config = Config("app/environment/environment.env")


def convert_toList_int(data):
    result = list()
    data = data.split(" ")
    for i in data:
        result.append(int(i))
    return result

def convert_toList_float(data):
    result = list()
    data = data.split(" ")
    for i in data:
        result.append(float(i))
    return result

def get_optimizer(config):
     optimizer_str = config
     optimizer_str = optimizer_str.replace("lr_schedule", "lr_schedule".upper()).replace("optimizer_momentum","optimizer_momentum".upper())
     return optimizer_str

def get_lr_schedule(config):
    lr_schedule_str = config
    lr_schedule_str = lr_schedule_str.replace("boundaries", "boundaries".upper()).replace("values","values".upper())
    return lr_schedule_str

class Settings(BaseSettings):
    environment: str = os.getenv("BUILD_ENVIRONMENT", "DEV")
    print("========= environment =========: ", environment)
    if environment != "":
        config = Config("app/environment/environment.{}env".format(environment))
    GPU = config("gpu", default=0)
    BATCH_SIZE = int(config("batch_size", default= 26))
    CONTEXT_MODEL = config('context_model', default= 'resnet18')
    BODY_MODEL = config('body_model', default='resnet18')
    LEARNING_RATE = float(config('learning_rate', default=0.01))
    WEIGHT_DECAY = float(config('weight_decay', default=5e-4))
    MODEL_WEIGHT_PATH = config('model_weight_path', default='./weights') 
    
def get_settings():
    return Settings()
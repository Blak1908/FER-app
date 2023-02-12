import os
from pydantic import BaseSettings
from starlette.config import Config


config  = Config("app/environment/environment.env")

class Setting(BaseSettings):
    environment: str = os.getenv("BUILD_ENVIRONMENT", "DEV")
    print("========= environment =========: ", environment)
    if environment != "":
        config = Config("app/environment/environment.{}env".format(environment))
    WIDTH = config("width", default=48)
    HEIGHT = config("height", default=48)
    CHANNELS = config("channels", default= 3)
    BATCH_SIZE = config("batch_size", default= 64)
    NUM_CLASSES = config("num_classes", default= 7)
    VERBOSE = config("verbose", default= 1)
    N = config("n", default= 3)
    INIT_FM_DIM = config("init_fm_dim", default= 64)
    MAXIMUM_NUMBER_ITERATIONS = config("maximum_number_iterations", default= 100)
    STEPS_PER_EPOCH = config("steps_per_epoch")
    VAL_STEPS_PER_EPOCH = config("val_steps_per_epoch")
    LOSS = config("loss")
    BOUNDARIES = config("boundaries")
    VALUES = config("values", default= [0.1, 0.01, 0.001])
    LR_SCHEDULE = config("lr_schedule")
    INITIALIZER = config("initializer")
    OPTIMIZER_MOMENTUM = config("optimizer_momentum", default= 0.9)
    OPTIMIZER_ADDITIONAL_METRICS = config("optimizer_additional_metrics", default= ["accuracy"])
    OPTIMIZER = config("optimizer")
    TENSORBOARD = config("tensorboard")
    CHECKPOINT = config("checkpoint")
    CALLBACKS = config("callbacks")





def get_setting():
    return Setting()
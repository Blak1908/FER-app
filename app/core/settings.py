import os
import tensorflow as tf
from pydantic import BaseSettings
from starlette.config import Config
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import initializers




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
    NAME = config("name", default="resnet")
    WIDTH = int(config("width", default=48))
    HEIGHT = int(config("height", default=48))
    CHANNELS = int(config("channels", default= 3))
    BATCH_SIZE = int(config("batch_size", default= 128))
    NUM_CLASSES = int(config("num_classes", default= 7))
    VERBOSE = int(config("verbose", default= 1))
    N = int(config("n", default= 3))
    INIT_FM_DIM = int(config("init_fm_dim", default= 64))
    MAXIMUM_NUMBER_ITERATIONS = int(config("maximum_number_iterations", default= 100))
    LOSS = config("loss", default= tf.keras.losses.CategoricalCrossentropy(from_logits=True))
    BOUNDARIES = convert_toList_int(config("boundaries", default= [32000, 48000]))
    VALUES = convert_toList_float(config("values", default= [0.1, 0.01, 0.001]))
    LR_SCHEDULE = eval(get_lr_schedule(config("lr_schedule", default = schedules.PiecewiseConstantDecay(BOUNDARIES, VALUES))))
    INITIALIZER = eval(config("initializer", default=initializers.HeNormal()))
    OPTIMIZER_MOMENTUM = float(config("optimizer_momentum", default= 0.9))
    OPTIMIZER_ADDITIONAL_METRICS = config("optimizer_additional_metrics", default="accuracy")
    OPTIMIZER =  eval(get_optimizer(config("optimizer", default= SGD(learning_rate=LR_SCHEDULE, momentum=OPTIMIZER_MOMENTUM))))
    TENSORBOARD = eval(config("tensorboard", default=TensorBoard(log_dir=os.path.join(os.getcwd(), "logs"),histogram_freq=1,write_images=True)))
    CHECKPOINT = eval(config("checkpoint",default= ModelCheckpoint(os.path.join(os.getcwd(), "model_checkpoint"),save_freq="epoch")))
    CALLBACKS = eval(config("callbacks", default=[TENSORBOARD,CHECKPOINT]))
    SHORTCUT_TYPE = config("shortcut_type", default='identity')
    MODEL_RESNET_PATH = config("model_resnet_path", default='weights/ResNet_model_weighs-iters150.h5')
    MODEL_ALEXNET_PATH =config("model_alexnet_path", default='weights/AlexNet_weighs_iters150.h5')
    MODEL_VGGNET_PATH = config("model_vggnet_path", default='weights/VGGNet_weights_iters150.h5')
    MODEL_SVM_PATH = config("model_svm_path", default="weights/svm_model.pkl")


def get_settings():
    return Settings()
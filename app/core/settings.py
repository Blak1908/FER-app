import os, sys
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
from pydantic import BaseSettings
from starlette.config import Config
import tensorflow as tf
from tensorflow.keras.optimizers import schedules, SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import initializers


class Settings(BaseSettings):
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
    LOSS = config("loss", default= tf.keras.losses.CategoricalCrossentropy(from_logits=True))
    BOUNDARIES = config("boundaries", default= [32000, 48000])
    VALUES = config("values", default= [0.1, 0.01, 0.001])
    LR_SCHEDULE = config("lr_schedule", default= schedules.PiecewiseConstantDecay(BOUNDARIES, VALUES))
    INITIALIZER = config("initializer", default= initializers.HeNormal())
    OPTIMIZER_MOMENTUM = config("optimizer_momentum", default= 0.9)
    OPTIMIZER_ADDITIONAL_METRICS = config("optimizer_additional_metrics", default= ["accuracy"])
    OPTIMIZER = config("optimizer", default= SGD(learning_rate=LR_SCHEDULE, momentum=OPTIMIZER_MOMENTUM))
    TENSORBOARD = config("tensorboard", default=TensorBoard(log_dir=os.path.join(os.getcwd(), "logs"),histogram_freq=1,write_images=True))
    CHECKPOINT = config("checkpoint",default= ModelCheckpoint(os.path.join(os.getcwd(), "model_checkpoint"),save_freq="epoch"))
    CALLBACKS = config("callbacks", default=[TENSORBOARD,CHECKPOINT])
    SHORTCUT_TYPE = config("shortcut_type", default='identity')
    MODEL_RESNET_PATH = config("model_resnet_path", default='weights/ResNet_model_weighs-iters150.h5')
    MODEL_ALEXNET_PATH =config("model_alexnet_path", default='weights/AlexNet_weighs_iters150.h5')
    MODEL_VGGNET_PATH = config("model_vggnet_path", default='weights/VGGNet_weights_iters150.h5')
    MODEL_SVM_PATH = config("model_svm_path", default="weights/svm_model.pkl")


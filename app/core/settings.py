import os
from pydantic import BaseSettings
from starlette.config import Config
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers.experimental import SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import initializers
config  = Config("app/environment/environment.env")

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
    LOSS = config("loss", default= CategoricalCrossentropy(from_logits=True))
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
    MODEL_RESNET = config("model_resnet", default='weights/resnet.h5')
    MODEL_ALEXNET =config("model_alexnet", default='weights/alexnet.h5')
    MODEL_VGGNET = config("model_vggnet", default='weights/vggnet.h5')




def get_setting():
    return Settings()
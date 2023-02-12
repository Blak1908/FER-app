import sys
import os 
import tensorflow as tf 

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
from app.core.network_models import model_ResNet

def _load(checkpoint_path):
    checkpoint = tf.train.Checkpoint()
    checkpoint.restore(checkpoint_path).assert_consumed()
    return checkpoint


if not os.path.exists('weights'):
    os.mkdir('weights')

def load_resnet_model(weights_path):
    model = model_ResNet.init_ResNet_model()
    model.load_weights(weights_path)
    return model

settings = get_setting()


model = load_resnet_model(settings.MODEL_RESNET)

model.summary()




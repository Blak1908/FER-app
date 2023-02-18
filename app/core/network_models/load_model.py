import sys
import os 

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
from app.core.network_models import model_SVM, model_AlexNet, model_ResNet, model_VGGNet

settings = get_setting()
WIDTH = settings.WIDTH
HEIGH = settings.HEIGHT
CHANNELS = settings.CHANNELS
NUM_CLASSES = settings.NUM_CLASSES

input_shape = (WIDTH, HEIGH, CHANNELS)
num_classes = NUM_CLASSES



def load_models(model_name):
    if model_name =='alexnet':
        return model_AlexNet(input_shape, num_classes)
    elif model_name == 'vggnet':
        return model_VGGNet(input_shape, num_classes)
    elif model_name == 'resnet':
        return model_ResNet.init_ResNet_model()
    return  model_SVM.SVM()


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

def load_models(model_name):
    if model_name =='alexnet':
        return model_AlexNet()
    elif model_name == 'vggnet':
        return model_VGGNet()
    elif model_name == 'resnet':
        return model_ResNet()
    return model_SVM()


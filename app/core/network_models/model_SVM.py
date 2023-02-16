import pickle
import sys

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
from app.core.network_models import model_AlexNet, model_ResNet, model_VGGNet




settings = get_setting()



svm_model_path = settings.MODEL_SVM_PATH



def SVM():
    # Combine the predictions of the three models
    with open(svm_model_path, 'rb') as f:
        svm_model = pickle.load(f)
    return svm_model
        
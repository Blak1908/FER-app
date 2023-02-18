import pickle, sys, os
from sklearn.externals import joblib

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
from app.core.network_models.load_model import load_models
from sklearn.svm import SVC


settings = get_setting()

MODE_SVM_PATH = settings.MODEL_SVM_PATH




def SVM():
    svm_model = SVC(kernel='linear')
    # Load the SVM model
    with open(MODE_SVM_PATH,'wb') as f:
        pickle.dump(svm_model,f)   
    return svm_model
        
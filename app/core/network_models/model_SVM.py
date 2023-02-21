import pickle, sys, os

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

from settings import Settings
from load_model import load_models
from sklearn.svm import SVC 


settings = Settings()

MODE_SVM_PATH = settings.MODEL_SVM_PATH




def SVM():
    svm_model = SVC(kernel='linear')
    # Load the SVM model
    with open(MODE_SVM_PATH,'wb') as f:
        pickle.dump(svm_model,f)   
    return svm_model
        
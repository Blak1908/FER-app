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
        
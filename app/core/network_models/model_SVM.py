from app.core.settings import Settings
import pickle
from sklearn.svm import SVC 


settings = Settings()

MODE_SVM_PATH = settings.MODEL_SVM_PATH


def SVM():
    # Load the SVM model
    with open(MODE_SVM_PATH, 'rb') as f:
        svm_model = pickle.load(f)
    return svm_model
        
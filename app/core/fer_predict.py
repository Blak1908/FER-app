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
from app.core.network_models.load_model import load_models


import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.svm import SVC

settings = get_setting()

MODEL_RESNET_PATH = settings.MODEL_RESNET_PATH
MODEL_ALEXNET_PATH = settings.MODEL_ALEXNET_PATH
MODEL_VGGNET_PATH  = settings.MODEL_VGGNET_PATH
MOODEL_SVM_PATH = settings.MODEL_SVM_PATH


label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

def predict(img_dir):
    # Load the model from MODEL PATH
    model_AlexNet = load_models('alexnet')
    model_VGGNet = load_models('vggnet')
    model_ResNet = load_models('resnet')
    model_SVM = load_models("")
    
    # Load model weights
    model_AlexNet.load_weights(MODEL_ALEXNET_PATH)
    model_VGGNet.load_weights(MODEL_VGGNET_PATH)
    model_ResNet.load_weights(MODEL_RESNET_PATH)
    model_SVM.load_weights(MOODEL_SVM_PATH)
    # Load and preprocess the test image
    img = image.load_img(img_dir, target_size=(48, 48))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Obtain predictions from the pre-trained models
    alexnet_pred = model_AlexNet.predict(img)
    vggnet_pred = model_VGGNet.predict(img)
    resnet_pred = model_ResNet.predict(img)
    
    # Concatenate the predictions
    combined_pred = np.concatenate((alexnet_pred, vggnet_pred, resnet_pred), axis=1)
    
    
    # Use the SVM model to predict the class label
    
    svm_pred = model_SVM.predict(combined_pred)
    
    predicted_label = label_dict[svm_pred[0]]
    return predicted_label
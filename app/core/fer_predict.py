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
from app.core.network_models import load_model


import numpy as np
from tensorflow.keras.preprocessing import image


label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

def predict(img_dir):
    # Load the SVM model from disk
    svm_model = load_model.load_svm_model()

    img = image.load_img(img_dir, target_size=(48, 48), color_mode='rgb')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.reshape(1, 48*48*3)  # Flatten the image to a 1D array

    # Make the prediction using the SVM model
    prediction = svm_model.predict(img)

    return label_dict[prediction[0]]
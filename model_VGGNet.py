import tensorflow as tf
tf.config.run_functions_eagerly(False)
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential


class VGGNet(Sequential):
  def __init__(self, input_shape, num_classes):
    super().__init__()

    self.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 3)))
    self.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    self.add(MaxPooling2D(pool_size=(2, 2)))

    self.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    self.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    self.add(MaxPooling2D(pool_size=(2, 2)))

    self.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    self.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    self.add(MaxPooling2D(pool_size=(2, 2)))

    self.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    self.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    self.add(MaxPooling2D(pool_size=(2, 2)))

    self.add(Flatten())
    self.add(Dense(4096, activation='relu'))
    self.add(Dense(4096, activation='relu'))
    self.add(Dense(4096, activation='relu'))
    self.add(Dense(4096, activation='relu'))
    self.add(Dense(1000, activation='relu'))
    self.add(Dense(1000, activation='relu'))
    self.add(Dense(num_classes, activation='softmax'))
    opt = SGD(lr=0.02)
    self.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

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
from app.core.settings import get_settings
settings = get_settings()
MODEL_VGGNET_PATH  = settings.MODEL_VGGNET_PATH


model = VGGNet((48,48,3),7)
model.load_weights(MODEL_VGGNET_PATH)

model.summary()
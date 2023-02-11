from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.models import Sequential
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
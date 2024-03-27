import pandas as pd
import numpy as np
from keras.utils import to_categorical
import keras

def LoadData():
    sign_mnist_train = np.array(pd.read_csv("sign_mnist_train.csv"))
    sign_mnist_test = np.array(pd.read_csv("sign_mnist_test.csv"))

    train_set = sign_mnist_train[:, 1:]
    train_labels = sign_mnist_train[:, 0]
    test_set = sign_mnist_test[:, 1:]
    test_labels = sign_mnist_test[:, 0]

    train_labels = to_categorical(train_labels, 24)
    test_labels = to_categorical(test_labels, 24)

    classes: dict[int, str] = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'K',
    10: 'L',
    11: 'M',
    12: 'N',
    13: 'O',
    14: 'P',
    15: 'Q',
    16: 'R',
    17: 'S',
    18: 'T',
    19: 'U',
    20: 'V',
    21: 'W',
    22: 'X',
    23: 'Y',
}

    # Data normalization
    train_set = train_set.astype('float32') / 255
    train_set = train_set.reshape(train_set.shape[0], 28, 28, 1)
    test_set = test_set.astype('float32') / 255
    test_set = test_set.reshape(test_set.shape[0], 28, 28, 1)

    # Data augmentation
    datagen = keras.ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    datagen.fit(train_set)

    return (train_set, train_labels), (test_set, test_labels), classes

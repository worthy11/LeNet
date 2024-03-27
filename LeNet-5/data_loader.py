import pandas as pd
import numpy as np
from keras.utils import to_categorical

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
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z',
}

sign_mnist_train = np.array(pd.read_csv("sign_mnist_train.csv"))
sign_mnist_test = np.array(pd.read_csv("sign_mnist_test.csv"))

train_set = sign_mnist_train[:, 1:]
train_labels = sign_mnist_train[:, 0]
test_set = sign_mnist_test[:, 1:]
test_labels = sign_mnist_test[:, 0]

train_set = train_set.astype('float32') / 255
train_set = train_set.reshape(train_set.shape[0], 28, 28, 1)
test_set = test_set.astype('float32') / 255
test_set = test_set.reshape(test_set.shape[0], 28, 28, 1)

# print(train_set.shape)
train_labels = to_categorical(train_labels, 26)
test_labels = to_categorical(test_labels, 26)

# print(sign_mnist_train)


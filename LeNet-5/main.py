import cv2
import numpy as np
import matplotlib.pyplot as plt
from lenet_keras import *

capture = cv2.VideoCapture(0)
img: cv2.Mat

while(True):
    _, img = capture.read()
    crop = img[100:400, 400:700]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (28, 28))

    sample = np.array(crop)
    sample = np.expand_dims(sample, axis=-1)
    sample = np.expand_dims(sample, axis=0)
    label = np.argmax(lenet.predict(x=sample, batch_size=1))

    plt.imshow(sample[0, :, :, :])
    plt.title(classes[label])
    plt.show()

    cv2.imshow('Main', img)
    cv2.waitKey(1)
from keras import layers, activations
import keras
import numpy as np
import random
from config import *

grid_size = GRID_SIZE
num_boxes = NUM_BOXES
num_classes = NUM_CLASSES
width = WIDTH
height = HEIGHT
batch_size = BATCH_SIZE
in_channels = IN_CHANNELS

yolov1 = keras.Sequential()
yolov1.add(layers.Input((height, width, in_channels), batch_size))
for x in ARCHITECTURE:
    if type(x) == tuple:
        yolov1.add(layers.Conv2D(filters=x[1], kernel_size=x[0], strides=x[2], padding=x[3]))

    elif type(x) == str:
        yolov1.add(layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))

yolov1.add(layers.Flatten())
yolov1.add(layers.Dense(496, activations.relu, False))
yolov1.add(layers.Dense(grid_size*grid_size*(num_classes+num_boxes*5), activations.relu, False))
yolov1.compile(optimizer='sgd', loss='mse')

rand = [[[[random.random() for x in range(in_channels)] for y in range(height)] for z in range(width)]]
rand = np.array(rand)
output = yolov1.predict(rand, batch_size=1)
print(output.shape)
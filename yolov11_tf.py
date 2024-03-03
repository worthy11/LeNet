from keras import layers, activations
import keras
import random
from config import *

grid_size = GRID_SIZE
num_boxes = NUM_BOXES
num_classes = NUM_CLASSES
width = WIDTH
height = HEIGHT
batch_size = BATCH_SIZE
in_channels = 3

yolov1 = keras.Sequential()
yolov1.add(layers.Input((in_channels, height, width), batch_size))
for x in ARCHITECTURE:
    if type(x) == tuple:
        yolov1.add(layers.Conv2D(x[1], x[0], x[2], x[3]))

    elif type(x) == str:
        yolov1.add(layers.MaxPool2D(2, 2, "valid"))

yolov1.add(layers.Flatten())
yolov1.add(layers.Dense(496, activations.relu, False))
yolov1.add(layers.Dense(grid_size*grid_size*(num_classes+num_boxes*5), activations.relu, False))
yolov1.compile(optimizer='sgd', loss='mse')

rand = [[[random.random() for x in range(width)] for y in range(height)] for z in range(in_channels)]
output = yolov1.predict(rand)
print(output.shape)
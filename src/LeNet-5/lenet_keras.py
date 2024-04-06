from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.callbacks import ReduceLROnPlateau
import numpy as np
from data_loader import *

class LeNet5Keras():
    def __init__(self, train_set, train_labels, test_set, test_labels, epochs=5, batch_size=64, learning_rate=0.00001):
        self.model = self.InitializeModel()
        self.train_set = train_set         
        self.train_labels = train_labels         
        self.test_set = test_set         
        self.test_labels = test_labels
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = ReduceLROnPlateau(monitor='val_accuracy',
                                                     patience=2, verbose=1,
                                                     factor=0.5, min_lr=learning_rate)
        
    
    def InitializeModel(self):
        lenet = Sequential()
        lenet.add(Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), padding='same', activation='tanh', use_bias=True))
        lenet.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        lenet.add(Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='same', activation='tanh', use_bias=True))
        lenet.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        lenet.add(Flatten())
        lenet.add(Dense(units=120, activation='tanh', use_bias=True))
        lenet.add(Dense(units=84, activation='tanh', use_bias=True))
        lenet.add(Dense(units=10, activation='softmax'))
        return lenet

    def TrainModel(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(self.train_set,
                                 self.train_labels,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_data = (self.test_set, self.test_labels),
                                 callbacks = [self.learning_rate])
        results = self.model.evaluate(self.test_set, self.test_labels)
        print('Loss: {}\nAccuracy: {}'.format(results[0], results[1]))
        return history
    
    def Predict(self, sample):
        predictions = self.model(inputs=sample)
        label = np.argmax(predictions)
        confidence = max(predictions[0]) / np.sum(predictions[0])
        return label, confidence

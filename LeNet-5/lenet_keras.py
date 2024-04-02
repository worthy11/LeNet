# import keras2onxx
import matplotlib.pyplot as plt
from data_loader import *
from arc import *

class LeNet5():
    def __init__(self, train_set, train_labels, test_set, test_labels, classes, epochs=5, batch_size=128, learning_rate=0.00001):
        self.model = InitializeModel()
        self.train_set = train_set         
        self.train_labels = train_labels         
        self.test_set = test_set         
        self.test_labels = test_labels
        self.classes = classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = ReduceLROnPlateau(monitor='val_accuracy',
                                                     patience=2, verbose=1,
                                                     factor=0.5, min_lr=learning_rate)

    def TrainModel(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(self.train_set, self.train_labels, batch_size=self.batch_size,
                                epochs=self.epochs, validation_data = (self.test_set, self.test_labels),
                                callbacks = [self.learning_rate])
        results = self.model.evaluate(self.test_set, self.test_labels)
        print('Loss: {}\nAccuracy: {}'.format(results[0], results[1]))

        return (history, results)
        
    def Predict(self, sample):
        predictions = self.model.predict(x=sample, batch_size=1, verbose=0)
        label = self.classes[np.argmax(predictions)]
        # plt.imshow(sample[0, :, :, 0])
        # plt.title(self.classes[label])
        # plt.show()
        return label
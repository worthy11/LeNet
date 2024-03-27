import keras
# import keras2onxx
import matplotlib.pyplot as plt
from data_loader import *
from arc import *

class LeNet5():
    def __init__(self, test_set, test_labels, train_set, train_labels, epochs=5, batch_size=4, learning_rate=0.1):
        self.model = InitializeModel()
        self.train_set = train_set         
        self.train_labels = train_labels         
        self.test_set = test_set         
        self.test_labels = test_labels
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def TrainModel(self):
        history = self.model.fit(x=self.train_set, y=self.train_labels, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.test_set, self.test_labels), verbose=1)
        results = self.model.evaluate(self.test_set, self.test_labels)
        print('Loss: {}\nAccuracy: {}'.format(results[0], results[1]))

        return (history, results)
        
    def Predict(self):
        sample = self.test_set[np.random.randint(0, len(self.test_set))]
        sample = np.expand_dims(sample, axis=0)
        label = np.argmax(self.model.predict(x=sample, batch_size=1))
        plt.imshow(sample[0, :, :, :])
        plt.title(classes[label])
        plt.show()
        key = input()
import keras
# import keras2onxx
import matplotlib.pyplot as plt
from data_loader import *
from model import *

lenet = InitializeModel()
batch_size = 4
epochs = 5

lenet.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])
history = lenet.fit(x=train_set, y=train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_set, test_labels), verbose=1)
results = lenet.evaluate(test_set, test_labels)
    
print('Loss: {}\nAccuracy: {}'.format(results[0],results[1]))

# while (True):
#     img = test_set[np.random.randint(0, len(test_set))]
#     img = np.expand_dims(img, axis=0)
#     label = np.argmax(lenet.predict(x=img, batch_size=1))
#     plt.imshow(img[0, :, :, :])
#     plt.title(classes[label])
#     plt.show()
#     key = input()
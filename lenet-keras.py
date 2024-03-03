from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import keras
import matplotlib.pyplot as plt
from time import process_time

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

y_train= to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

lenet = keras.Sequential()

lenet.add(keras.layers.Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), padding='same', activation='tanh', use_bias=True))
lenet.add(keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
lenet.add(keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='same', activation='tanh', use_bias=True))
lenet.add(keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
lenet.add(keras.layers.Flatten())
lenet.add(keras.layers.Dense(units=120, activation='tanh', use_bias=True))
lenet.add(keras.layers.Dense(units=84, activation='tanh', use_bias=True))
lenet.add(keras.layers.Dense(units=10, activation='softmax', use_bias=True))

batch_size = 4
epochs = 5

figure, axis = plt.subplots(3, 2)
axis[0,0].set_title('Loss comparison')
axis[0,1].set_title('Accuracy comparison')
x = [_+1 for _ in range(epochs)]
time_x = [_ for _ in range(2, 10)]
time_y = []
row = 0
lenet.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])
while batch_size < 513:
    if batch_size > 32:
        row = 1
    
    start = process_time()
    history = lenet.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test,y_test), verbose=1)
    results = lenet.evaluate(x_test, y_test)
    end = process_time()
    
    axis[row,0].plot(x, history.history['loss'], label='Loss (size {})'.format(batch_size))
    axis[row,0].plot(x, history.history['val_loss'], label='Vloss (size {})'.format(batch_size))
    axis[row,1].plot(x, history.history['accuracy'], label='Acc (size {})'.format(batch_size))
    axis[row,1].plot(x, history.history['val_accuracy'], label='Vacc (size {})'.format(batch_size))
    
    time_y.append(end-start)
    batch_size *= 2
axis[0,0].legend()
axis[0,1].legend()
axis[1,0].legend()
axis[1,1].legend()
axis[2,0].scatter(time_x, time_y)
axis[2,0].set_title('CPU time')
plt.show()
print('Loss: {}\nAccuracy: {}'.format(results[0],results[1]))
# import keras

# def InitializeModel():
#     lenet = keras.Sequential()

#     lenet.add(keras.Conv2D(75, (3,3), strides=1, padding='same', activation='relu', input_shape = (28,28,1)))
#     lenet.add(keras.BatchNormalization())
#     lenet.add(keras.MaxPool2D((2,2), strides=2, padding='same'))

#     lenet.add(keras.Conv2D(50, (3,3), strides=1, padding='same', activation='relu'))
#     lenet.add(keras.Dropout(0.2))
#     lenet.add(keras.BatchNormalization())
#     lenet.add(keras.MaxPool2D((2,2), strides=2, padding='same'))

#     lenet.add(keras.Conv2D(25, (3,3), strides=1, padding = 'same', activation='relu'))
#     lenet.add(keras.BatchNormalization())
#     lenet.add(keras.MaxPool2D((2,2), strides=2, padding='same'))

#     lenet.add(keras.Flatten())
#     lenet.add(keras.Dense(units=512, activation='relu'))
#     lenet.add(keras.Dropout(0.3))
#     lenet.add(keras.Dense(units=26, activation='softmax'))
    
#     # lenet.summary()

#     return lenet
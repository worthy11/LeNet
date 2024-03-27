import keras

def InitializeModel():
    lenet = keras.Sequential()

    lenet.add(keras.layers.Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), padding='same', activation='tanh', use_bias=True))
    lenet.add(keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    lenet.add(keras.layers.Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='same', activation='tanh', use_bias=True))
    lenet.add(keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    lenet.add(keras.layers.Flatten())
    lenet.add(keras.layers.Dense(units=120, activation='tanh', use_bias=True))
    lenet.add(keras.layers.Dense(units=84, activation='tanh', use_bias=True))
    lenet.add(keras.layers.Dense(units=26, activation='softmax', use_bias=True))

    return lenet
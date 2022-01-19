import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, LeakyReLU, ConvLSTM2D
import random

import config as cfg
import data_process
from keras_sequence import KerasSequence

def kernel(x):
    return (x, x)

def create_model(shape:tuple):
    inputs = keras.layers.Input(shape=shape)
    x = ConvLSTM2D(48, kernel(5), kernel(2))(inputs)
    x = Conv2D(48, kernel(3), kernel(1))(x)
    #x = Conv2D(48, kernel(3), kernel(1))(x)
    #x = ConvLSTM2D(48, kernel(3), kernel(1))(x)
    x = MaxPool2D()(x)
    x = Dense(32, activation="tanh")(x)
    x = Dense(4, activation="sigmoid")(x)
    return keras.models.Model(inputs, x)


if __name__ == '__main__':
    modele = create_model((2, cfg.height, cfg.width, 3))
    modele.summary()
    modele.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')
    es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, restore_best_weights=True)
    entrees = data_process.load_data()
    random.shuffle(entrees)
    train = KerasSequence(entrees[:int(len(entrees)*0.8)])
    validation = KerasSequence(entrees[int(len(entrees)*0.8):int(len(entrees)*0.98)])
    test = KerasSequence(entrees[int(len(entrees)*0.98):])
    modele.fit(train, validation_data=validation, epochs=40, callbacks=[es])
    breakpoint()


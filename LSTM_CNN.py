import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, LeakyReLU, ConvLSTM2D
import random

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
import data_process
from keras_sequence import KerasSequence

def kernel(x):
    return (x, x)

def create_model(shape:tuple):
    inputs = keras.layers.Input(shape=shape)
    x = ConvLSTM2D(48, kernel(5), kernel(2), return_sequences=True)(inputs)
    #x = ConvLSTM2D(48, kernel(3), kernel(1), return_sequences=True)(x)
    x = ConvLSTM2D(48, kernel(3), kernel(1), return_sequences=False)(x)
    x = MaxPool2D()(x)
    x = Conv2D(48, kernel(3), kernel(1), activation="tanh")(x)
    #x = Conv2D(48, kernel(3), kernel(1), activation="tanh")(x)
    x = Dense(32, activation="tanh")(x)
    x = Dense(4, activation="sigmoid")(x)
    return keras.models.Model(inputs, x)


def plot_y(y):
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(y[:,:,0])
    axarr[0,0].set_title('Tête')
    axarr[0,1].imshow(y[:,:,1])
    axarr[0,1].set_title('Maillot')
    axarr[1,0].imshow(y[:,:,2])
    axarr[1,0].set_title('Bras droit')
    axarr[1,1].imshow(y[:,:,3])
    axarr[1,1].set_title('Bras gauche')
    plt.show()


if __name__ == '__main__':
    modele = create_model((2, cfg.height, cfg.width, 3))
    modele.summary()
    modele.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')
    es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, restore_best_weights=True)
    entrees = data_process.load_data()
    random.seed(42)
    random.shuffle(entrees)
    train = KerasSequence(entrees[:int(len(entrees)*0.8)])
    validation = KerasSequence(entrees[int(len(entrees)*0.8):int(len(entrees)*0.98)])
    test = entrees[int(len(entrees)*0.98):]
    modele.fit(train, validation_data=validation, epochs=40, callbacks=[es])
    #modele = keras.models.load_model('modele.h5')

    modele.save('modele.h5')
    for e in test:
        x = e.x()
        y = e.y()
        pred = modele(np.array([x]))
        plt.imshow(x[0])
        plt.show()
        plot_y(y)
        plot_y(pred[0])

        

    breakpoint()


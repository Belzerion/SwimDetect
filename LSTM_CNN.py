import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, LeakyReLU, ConvLSTM2D, Concatenate, Reshape
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

    conv_1 = Conv2D(64, kernel(3), kernel(1), activation="tanh")
    conv_2 = Conv2D(96, kernel(3), kernel(1), activation="tanh")
    conv_3 = Conv2D(128, kernel(3), kernel(1), activation="tanh")

    x = conv_1(inputs[:, 0])
    x = conv_2(x)
    x = conv_3(x)
    x = MaxPool2D()(x)

    y = conv_1(inputs[:, 1])
    y = conv_2(y)
    y = conv_3(y)
    y = MaxPool2D()(y)

    x = Reshape((1, x.shape[1], x.shape[2], x.shape[3]))(x)
    y = Reshape((1, y.shape[1], y.shape[2], y.shape[3]))(y)

    x = Concatenate(axis=1)([x, y])
    
    x = ConvLSTM2D(64, kernel(3), kernel(1), return_sequences=True)(x)
    x = ConvLSTM2D(64, kernel(3), kernel(1), return_sequences=True)(x)
    x = ConvLSTM2D(64, kernel(3), kernel(1), return_sequences=False)(x)
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
        print(e.fichier_2)
        x = e.x()
        y = e.y()
        pred = modele(np.array([x]))
        plt.imshow(x[1])
        plt.show()
        plot_y(y)
        plot_y(pred[0])

        

    breakpoint()


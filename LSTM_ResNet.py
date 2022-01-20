import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, LeakyReLU, ConvLSTM2D, Concatenate, Reshape
import random
from tensorflow.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint

import config as cfg
import data_process
from keras_sequence import KerasSequence

def kernel(x):
    return (x, x)

class UnFreezeWeight(tf.keras.callbacks.Callback):
    def __init__(self, freeze_before_epoch):
        super().__init__()
        self.freeze_before_epoch = freeze_before_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if self.freeze_before_epoch != epoch:
            return

        # Unfreeze all weight.
        self.model.make_train_function(force=True)
        print('set trainable to True.')
        for layer in self.model.layers:
            layer.trainable = True

    


def create_model(shape:tuple):
    inputs = keras.layers.Input(shape=shape)
    """
    conv_1 = Conv2D(64, kernel(3), kernel(1), activation="tanh")
    conv_2 = Conv2D(96, kernel(3), kernel(1), activation="tanh")
    conv_3 = Conv2D(128, kernel(3), kernel(1), activation="tanh")
    """
    
    resnetCnn = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor = None,
        input_shape=inputs[:,0].shape[1:],
        pooling=None,
    )
    resnetCnn.trainable = False
    index = 0
    for layer in resnetCnn.layers:
        if layer.name == 'conv3_block1_1_conv':
            index = resnetCnn.layers.index(layer)
            break
    
    model = tf.keras.models.Model(resnetCnn.input, resnetCnn.layers[index-1].output)
    """
    index = 0
    for layer in resnetCnn.layers:
        if layer.name == 'conv3_block1_1_conv':
            index = resnetCnn.layers.index(layer)
            break
    del resnetCnn.layers[index:]
    
    while len(resnetCnn.layers) > index:
        resnetCnn._layers.pop()    
        
    resnetCnn.trainable = False
    """
    """
    x = conv_1(inputs[:, 0])
    x = conv_2(x)
    x = conv_3(x)
    x = MaxPool2D()(x)

    y = conv_1(inputs[:, 1])
    y = conv_2(y)
    y = conv_3(y)
    y = MaxPool2D()(y)
    """
    x = model(inputs[:, 0])
    #x = MaxPool2D()(x)
    
    y = model(inputs[:, 1])
    #y = MaxPool2D()(y)
    x = Reshape((1, x.shape[1], x.shape[2], x.shape[3]))(x)
    y = Reshape((1, y.shape[1], y.shape[2], y.shape[3]))(y)

    x = Concatenate(axis=1)([x, y])
    #x = ConvLSTM2D(64, kernel(3), kernel(1), return_sequences=True)(x)
    x = ConvLSTM2D(64, kernel(3), kernel(1), return_sequences=True, activation = LeakyReLU())(x)
    x = ConvLSTM2D(64, kernel(3), kernel(1), return_sequences=False, activation = LeakyReLU())(x)
    x = Dense(32, activation=LeakyReLU())(x)
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
    cb = UnFreezeWeight(4)
    entrees = data_process.load_data()
    random.seed(42)
    random.shuffle(entrees)
    train = KerasSequence(entrees[:int(len(entrees)*0.8)])
    validation = KerasSequence(entrees[int(len(entrees)*0.8):int(len(entrees)*0.98)])
    test = entrees[int(len(entrees)*0.98):]
    modele.fit(train, validation_data=validation, epochs=10, callbacks=[es, cb])
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


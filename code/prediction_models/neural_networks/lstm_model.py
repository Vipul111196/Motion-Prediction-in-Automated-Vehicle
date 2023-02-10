import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from keras.layers import BatchNormalization
from tensorflow.keras.layers import *
import numpy as np

def LSTM_keras_model(xTrain, xTest, yTrain, yTest):
    # flatten the input and output
    n_input = np.shape(xTrain)[1] * np.shape(xTrain)[2]
    xTrain = np.reshape(xTrain, (np.shape(xTrain)[0], n_input))
    n_output = np.shape(yTrain)[1] * np.shape(yTrain)[2]
    yTrain = np.reshape(yTrain, (np.shape(yTrain)[0], n_output))

    xTest = np.reshape(xTest, (np.shape(xTest)[0], n_input))
    yTest = np.reshape(yTest, (np.shape(yTest)[0], n_output))

    model = Sequential()
    model.add(Bidirectional(LSTM(100, activation='relu', input_shape=(100,1)))) #elu
    model.add(Bidirectional(LSTM(50, dropout=0.5)))
    model.add(BatchNormalization(momentum=0.6))
    model.add(Dense(n_output))
    #modell.compile(loss='mean_squared_error', optimizer='adam')

    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), 
              optimizer=tf.keras.optimizers.Adam(lr=0.01),  # lr = 0.1 or 0.001
              metrics=['mae'])

    history = model.fit(xTrain, yTrain, epochs=100, verbose=1, validation_data=(xTest, yTest))

    # save the trained model
    filepath = 'LSTM_model.h5'
    model.save(filepath)

    # load trained model
    model = keras.models.load_model('LSTM_model.h5')

    # Training and validation loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')

    return model

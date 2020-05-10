"""
This Module contains funciton used to help evaluate X-Position  O-PPAC data
@author: Ted Yoo
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, metrics, Model
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR 
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Takes (9,132,1011) dataset from Yassid and compiles it into a (9099,132) set. 
def data_compile(histdata, x_pos):
    data = []
    labels = []
    for i in range(histdata.shape[0]):
    	#Starts at 1 because first event of each x-position has no charge recorded.
        for j in range(1,histdata.shape[2]):
            data.append(histdata[i,0:132,j])
            labels.append(x_pos[i])
    return(np.array(data),np.array(labels))

#Splits all data into Train, Validation and Test based on input percentages of test_split and val_split
def data_split(dataset,labels,test_split, val_split):
    temp_data, test_data, temp_labels, test_labels = train_test_split(dataset, labels, test_size=test_split,shuffle=True)
    train_data, val_data, train_labels, val_labels = train_test_split(temp_data, temp_labels, test_size=val_split,shuffle=True)
    return (train_data, val_data, test_data, train_labels, val_labels, test_labels)

#Scales Train, Validation and Test based on specified scaler_type
def TVT_scaler(scaler_type, train, val, test):
    scaler_type.fit(train)
    train = scaler_type.transform(train)
    val = scaler_type.transform(val)
    test = scaler_type.transform(test)
    return (train, val, test)

#Scales Train and Test based on specified scaler_type
def TT_scaler(scaler_type, train, test):
    scaler_type.fit(train)
    train = scaler_type.transform(train)
    test = scaler_type.transform(test)
    return (train, test)    

#Trains Fully Connected Neural Network Model, Evaluates based on Test, Validation, Train inputs, and Predicts
def train_FCNN_model(trainData, trainLabels, valData, valLabels, testData, testLabels): 
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape = [132]))
    model.add(layers.Dense(256, activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation = 'linear'))
    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001), 
              loss = 'mse', 
              metrics=['mae', 'mse'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    history = model.fit(trainData, trainLabels,
                    epochs=100,
                    batch_size = 32,
                    callbacks = [callback],
                    validation_data = (valData,valLabels)
                    )
    result = model.evaluate(testData, testLabels, verbose=2)
    predictions = model.predict(testData)
    return(history, result, predictions)

#Train SVR Model and Predict 
def SVR_Model(train_data, train_labels, test_data):
    Model = SVR()
    Model.fit(train_data, train_labels.ravel())
    predictions = Model.predict(test_data)
    return predictions
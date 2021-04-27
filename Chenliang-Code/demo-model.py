#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def Model(index_step_length, feature_number, cell_type, loss_type, optimizer_type):
        model = keras.Sequential()
        if cell_type == 0:
            model.add(keras.layers.SimpleRNN(64, kernel_initializer='glorot_uniform', batch_input_shape=(None, index_step_length, feature_number), return_sequences=True, name='Encoder'))
            model.add(keras.layers.SimpleRNN(64, kernel_initializer='glorot_uniform', return_sequences=True, name='Decoder'))
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(feature_number)))
                    
        if cell_type == 1:
            model.add(keras.layers.LSTM(64, kernel_initializer='glorot_uniform', batch_input_shape=(None, index_step_length, feature_number), return_sequences=True, name='Encoder'))           
            model.add(keras.layers.LSTM(64, kernel_initializer='glorot_uniform', return_sequences=True, name='Decoder'))
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(feature_number)))
        
        if cell_type == 2:
            model.add(keras.layers.GRU(64, kernel_initializer='glorot_uniform', batch_input_shape=(None, index_step_length, feature_number), return_sequences=True, name='Encoder'))
            model.add(keras.layers.GRU(64, kernel_initializer='glorot_uniform', return_sequences=True, name='Decoder'))
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(feature_number)))
        
        model.compile(loss=loss_type,optimizer=optimizer_type,metrics=['accuracy'])
        model.build()
        return model
    
#Split and reshape the data set by step_size , use min-max or stanrdardlize method to rescale the data
def Splitting_dataset(data, step_size, scale=True, scaler_type=MinMaxScaler):
        l = len(data) 
        data = scaler_type().fit_transform(data)
        Xs = []
        Ys = []
        for i in range(0, (len(data) - window_size)):
            Xs.append(data[i:i+step_size])
            Ys.append(data[i:i+step_size])
        train_x, test_x, train_y, test_y = [np.array(x) for x in train_test_split(Xs, Ys)]
        assert train_x.shape[2] == test_x.shape[2] == (data.shape[1] if (type(data) == np.ndarray) else len(data))
        return  (train_x.shape[2], train_x, train_y, test_x, test_y)


        
def Summary(model):
    print(model.summary())

def trainloss(model,data_X):
    x_train_pred = model.predict(data_X)
    train_loss = np.square(np.mean(np.abs(x_train_pred - data_X), axis=1))

def testloss(model, data_XX):
    x_test_pred = model.predict(data_XX)
    test_loss = np.square(np.mean(np.abs(x_test_pred - data_XX),axis=1))

def get_threshold(train_loss):
    threshold = np.max(train_loss)
    print (threshold)

if __name__ == '__main__':
    data = pd.read_csv('c172_file_1.csv')
    step_size= 10
    epoch = 100
    batch = 100
    index_step_length = 10
    loss_type = "mse"
    optimizer_type = "adam"
    cells = [0,1,2]
    labels, X, Y, XX, YY = Splitting_dataset(data, step_size)
    for cell_type in cells:
        demo = Model(index_step_length, labels, cell_type, loss_type, optimizer_type)
        Summary(demo)
        history = demo.fit(x=X, y=Y, validation_data=(XX, YY), epochs=epoch, batch_size=batch, shuffle=True).history
    
        fig, loss_validation= plt.subplots(figsize=(14,8), dpi=80)
        loss_validation.plot(history['loss'],'b',label = 'Train',linewidth=2)
        loss_validation.plot(history['val_loss'],'r',label = 'Validation',linewidth=2)
        loss_validation.set_xlabel('Epoch')
        loss_validation.set_ylabel('Loss(mse)')
        loss_validation.legend(loc='center right')
        loss_validation.set_title("Loss graph " + str(cell_type))
        plt.savefig('Loss'  + str(cell_type) +".png", format="PNG" )

        fig, accuracy_validation= plt.subplots(figsize=(14,8), dpi=80)
        accuracy_validation.plot(history['accuracy'],'b',label = 'Train',linewidth=2)
        accuracy_validation.plot(history['val_accuracy'],'r',label = 'Validatioin',linewidth=2)
        accuracy_validation.set_xlabel('Epoch')
        accuracy_validation.set_ylabel('Accuracy(mse)')
        accuracy_validation.legend(loc='center right')
        accuracy_validation.set_title("Accuracy graph " + str(cell_type))
        plt.savefig('Accuracy' + str(cell_type) +".png", format="PNG" )
    
    








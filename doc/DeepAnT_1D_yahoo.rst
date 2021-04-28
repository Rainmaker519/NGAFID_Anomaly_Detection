.. code:: ipython3

    # Things to be included
    # 1. visualization tools for optimization
    # 2. visualization tools for plotting actual and predicted sequence, and anomaly points
    # 3. Computation cost calculation
    # 4. Warnings / Errors?

.. code:: ipython3

    import numpy as np
    from numpy import array
    import random
    from random import randint
    import os
    import matplotlib.pyplot as plt
    import pandas as pd
    import sklearn.model_selection
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Conv1D, Flatten, Activation, MaxPooling1D, Dropout
    from keras.optimizers import SGD
    import pdb

.. code:: ipython3

    #model will be trained on GPU 1, I don't need it now.
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"

.. code:: ipython3

    def anomaly_detector(prediction_seq, ground_truth_seq):
        
        """ Anomaly detector
        
        This is the method to determin whether this is an anomaly outlier.
        Enclidean distance is used to calculate the number here.
        
        parameter
        ----
        prediction_seq: list.
        ground_truth_seq: array.
        
        return
        ----
        final_list: array of boolean.
        
        """
        # calculate Euclidean between actual seq and predicted seq
        if len(prediction_seq) != len(ground_truth_seq):
            print("Error! You did not have same length of sequence!")
            return
        else:
            final_list = []
            for i in range(len(prediction_seq)):
                dist = np.linalg.norm(ground_truth_seq[i][0] - prediction_seq[i])
                if (abs(dist) > anm_det_thr):
                    final_list.append(True)  # anomaly
                else:
                    final_list.append(False) # normal
            return np.array(final_list)

.. code:: ipython3

    """Hyperparameters
    
    List down all the hyperparamters and the explianation.
    """
    
    w = 10                 # History window (number of time stamps taken into account) 
                             # i.e., filter(kernel) size       
    p_w = 1                # Prediction window (number of time stampes required to be 
                             # predicted)
    n_features = 1           # Univariate time series
    
    kernel_size = 2          # Size of filter in conv layers
    num_filt_1 = 32          # Number of filters in first conv layer
    num_filt_2 = 32          # Number of filters in second conv layer
    num_nrn_dl = 40          # Number of neurons in dense layer
    num_nrn_ol = p_w         # Number of neurons in output layer
    
    conv_strides = 1
    pool_size_1 = 2          # Length of window of pooling layer 1
    pool_size_2 = 2          # Length of window of pooling layer 2
    pool_strides_1 = 2       # Stride of window of pooling layer 1
    pool_strides_2 = 2       # Stride of window of pooling layer 2
    
    epochs = 100
    dropout_rate = 0.5       # Dropout rate in the fully connected layer
    learning_rate = 0.01  
    anm_det_thr = 0.5        # Threshold for classifying anomaly (0.5~0.8)

.. code:: ipython3

    """Data loading
    
    Load the data and normalize the value.
    Separate the is_anomaly data out from the dataset.
    Plot the overview of the dataset
    """
    
    yahoo = pd.read_csv('/Users/fermat/Desktop/RIT_Class/2021 Spring/Data_Science_project/paper_dataset/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_1.csv')
    plt.plot(yahoo['value'])
    plt.title('yahoo')
    plt.ylabel('value')
    plt.xlabel('index')
    plt.legend(['yahoo'], loc='upper right')
    plt.figure(figsize=(100,10))
    plt.show()
    print(yahoo.head())
    norm_data = list(yahoo['value'])
    yahoo['value'] = [(float(i)-min(norm_data))/(max(norm_data)-min(norm_data)) for i in norm_data]
    true_label = yahoo['is_anomaly'].copy()
    yahoo = yahoo.drop(columns=['is_anomaly'])
    yahoo.head()




.. image:: DeepAnT_1D_yahoo_files/DeepAnT_1D_yahoo_5_0.png



.. parsed-literal::

    <Figure size 7200x720 with 0 Axes>


.. parsed-literal::

       timestamp     value  is_anomaly
    0          1  0.000000           0
    1          2  0.091758           0
    2          3  0.172297           0
    3          4  0.226219           0
    4          5  0.176358           0




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>timestamp</th>
          <th>value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>0.116573</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>0.218893</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>0.287399</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>0.224053</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    """Data preprocessing"""
    
    # split a univariate sequence into samples
    def split_sequence(sequence):
        """
        In the model,
        Sample batch: w data points(from time (t-w) to (t)).
        batch label: p_w(prediction window) data points(at time (t+p_w)).
        
        """
        X, y = [], []
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + w
            out_end_ix = end_ix + p_w
            # check if we are beyond the sequence
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
            
        return array(X), array(y)
          
    
    # define input sequence
    raw_seq = list(yahoo['value'])
    
    # split into samples
    batch_sample, batch_label = split_sequence(raw_seq)
    print(batch_sample)
    print(batch_sample.shape)
    
    
    # 2. reshape from [samples, timesteps] into [samples, timesteps, features]
    # need to convert batch into 3D tensor of the form [batch_size, input_seq_len, n_features]
    
    batch_sample = batch_sample.reshape((batch_sample.shape[0], batch_sample.shape[1], n_features))


.. parsed-literal::

    [[0.         0.11657333 0.21889332 ... 0.08693264 0.16935357 0.10802149]
     [0.11657333 0.21889332 0.28739895 ... 0.16935357 0.10802149 0.12332587]
     [0.21889332 0.28739895 0.22405289 ... 0.10802149 0.12332587 0.        ]
     ...
     [0.05579008 0.06745122 0.26040117 ... 0.22281139 0.20285787 0.17484625]
     [0.06745122 0.26040117 0.04549539 ... 0.20285787 0.17484625 0.25083827]
     [0.26040117 0.04549539 0.04189874 ... 0.17484625 0.25083827 0.20576891]]
    (1410, 10)


.. code:: ipython3

    """Generate model for predictor"""
    model = Sequential()
    
    # Convolutional Layer #1
    # Computes 32 features using a 1D filter(kernel) of with w with ReLU activation. 
    # Padding is added to preserve width.
    # Input Tensor Shape: [batch_size, w, 1] / batch_size = len(batch_sample)
    # Output Tensor Shape: [batch_size, w, num_filt_1] (num_filt_1 = 32 feature vectors)
    model.add(Conv1D(filters=num_filt_1,
                     kernel_size=kernel_size,
                     strides=conv_strides,
                     padding='valid',
                     activation='relu',
                     input_shape=(w, n_features)))
    
    # Pooling Layer #1
    # First max pooling layer with a filter of length 2 and stride of 2
    # Input Tensor Shape: [batch_size, w, num_filt_1]
    # Output Tensor Shape: [batch_size, 0.5 * w, num_filt_1]
    
    model.add(MaxPooling1D(pool_size=pool_size_1)) 
                        #  strides=pool_strides_1, 
                        #  padding='valid'))
    
    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 0.5 * w, 32]
    # Output Tensor Shape: [batch_size, 0.5 * w, num_filt_1 * num_filt_2]
    model.add(Conv1D(filters=num_filt_2,
                     kernel_size=kernel_size,
                     strides=conv_strides,
                     padding='valid',
                     activation='relu'))
    
    # Max Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 0.5 * w, num_filt_1 * num_filt_2]
    # Output Tensor Shape: [batch_size, 0.25 * w, num_filt_1 * num_filt_2]
    model.add(MaxPooling1D(pool_size=pool_size_2))
                        #  strides=pool_strides_2, 
                        #  padding='valid'
              
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 0.25 * w, num_filt_1 * num_filt_2]
    # Output Tensor Shape: [batch_size, 0.25 * w * num_filt_1 * num_filt_2]
    model.add(Flatten())
    
    # Dense Layer (Output layer)
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 0.25 * w * num_filt_1 * num_filt_2]
    # Output Tensor Shape: [batch_size, 1024]
    model.add(Dense(units=num_nrn_dl, activation='relu'))  
    
    # Dropout
    # Prevents overfitting in deep neural networks
    model.add(Dropout(dropout_rate))
    
    # Output layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, p_w]
    model.add(Dense(units=num_nrn_ol))
    
    # Summarize model structure
    model.summary()


.. parsed-literal::

    Model: "sequential_9"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv1d_18 (Conv1D)           (None, 9, 32)             96        
    _________________________________________________________________
    max_pooling1d_18 (MaxPooling (None, 4, 32)             0         
    _________________________________________________________________
    conv1d_19 (Conv1D)           (None, 3, 32)             2080      
    _________________________________________________________________
    max_pooling1d_19 (MaxPooling (None, 1, 32)             0         
    _________________________________________________________________
    flatten_9 (Flatten)          (None, 32)                0         
    _________________________________________________________________
    dense_16 (Dense)             (None, 40)                1320      
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 40)                0         
    _________________________________________________________________
    dense_17 (Dense)             (None, 1)                 41        
    =================================================================
    Total params: 3,537
    Trainable params: 3,537
    Non-trainable params: 0
    _________________________________________________________________


.. code:: ipython3

    '''configure model'''
    model.compile(optimizer='adam', 
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    
    # In paper -> mean_absolute_error
    
    # sgd = keras.optimizers.SGD(lr=learning_rate, 
    #                          decay=1e-6, 
    #                          momentum=0.9, 
    #                          nesterov=True)
    # model.compile(optimizer='sgd', 
    #               loss='mean_absolute_error', 
    #               metrics=['accuracy'])

.. code:: ipython3

    '''Training'''
    history = model.fit(batch_sample, batch_label, validation_split=0.2, epochs=epochs, batch_size=15, verbose=0).history
    # model_fit = model.fit(X_train,
    #                       y_train,
    #                       epochs=epochs,
    #                       verbose=1)

.. code:: ipython3

    fig, loss_validation= plt.subplots(figsize=(14,8), dpi=80)
    loss_validation.plot(history['loss'],'b',label = 'Train',linewidth=2)
    loss_validation.plot(history['val_loss'],'r',label = 'Validation',linewidth=2)
    loss_validation.set_xlabel('Epoch')
    loss_validation.set_ylabel('Loss(mse)')
    loss_validation.legend(loc='center right')
    plt.savefig('Loss')
    
    fig, accuracy_validation= plt.subplots(figsize=(14,8), dpi=80)
    accuracy_validation.plot(history['accuracy'],'b',label = 'Train',linewidth=2)
    accuracy_validation.plot(history['val_accuracy'],'r',label = 'Validatioin',linewidth=2)
    accuracy_validation.set_xlabel('Epoch')
    accuracy_validation.set_ylabel('Accuracy(mse)')
    accuracy_validation.legend(loc='center right')
    plt.savefig('Accuracy')



.. image:: DeepAnT_1D_yahoo_files/DeepAnT_1D_yahoo_10_0.png



.. image:: DeepAnT_1D_yahoo_files/DeepAnT_1D_yahoo_10_1.png


.. code:: ipython3

    """Save Weights (DeepAnT)"""
    # save it to disk so we can load it back up anytime
    model.save_weights('yahoo_DeepAnT.h5')

.. code:: ipython3

    """Testing with random interval(DeepAnT)"""
    # Set number of test sequences 
    n_test_seq = 20
    
    # Split a univariate sequence into samples
    def generate_test_batch(raw_seq, n_test_seq):
      # Sample a portion of the raw_seq randomly
        ran_ix = random.randint(0,len(raw_seq) - n_test_seq * w - n_test_seq * p_w)
        raw_test_seq = array(raw_seq[ran_ix:ran_ix + n_test_seq * w +  n_test_seq * p_w])
        batch_test_seq, batch_test_label = list(), list()
        ix = ran_ix
        for i in range(n_test_seq):
            # gather input and output parts of the pattern
            seq_x = raw_seq[ix : ix+w],
            seq_y = raw_seq[ix+w : ix+w+p_w]
            ix = ix+w+p_w
            batch_test_seq.append(seq_x)
            batch_test_label.append(seq_y)
        return array(batch_test_seq), array(batch_test_label)
    
    batch_test_seq, batch_test_label = generate_test_batch(list(yahoo['value']), n_test_seq)
    batch_test_seq = batch_test_seq.reshape((batch_test_seq.shape[0], w, n_features))
    batch_test_label = batch_test_label.reshape((batch_test_label.shape[0], p_w))
    
    # Returns the loss value & metrics values for the model in test mode
    model.evaluate(x=batch_test_seq,
                   y=batch_test_label,
                   verbose=1)


.. parsed-literal::

    1/1 [==============================] - 0s 1ms/step - loss: 0.0018 - accuracy: 0.0000e+00




.. parsed-literal::

    [0.0018367994343861938, 0.0]



.. code:: ipython3

    """Predicting random intervals (DeepAnT)"""
    # Build model 
    model = Sequential()
    model.add(Conv1D(filters=num_filt_1,
                     kernel_size=kernel_size,
                     strides=conv_strides,
                     padding='valid',
                     activation='relu',
                     input_shape=(w, n_features)))
    model.add(MaxPooling1D(pool_size=pool_size_1)) 
    model.add(Conv1D(filters=num_filt_2,
                     kernel_size=kernel_size,
                     strides=conv_strides,
                     padding='valid',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size_2))
    model.add(Flatten())
    model.add(Dense(units=num_nrn_dl, activation='relu')) 
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=num_nrn_ol))
    
    # Load the model's saved weights.
    model.load_weights('aws_DeepAnT.h5')
              
    # Sample a portion of the raw_seq randomly
    # 1. Choose 
    ran_ix = random.randint(1,len(raw_seq) - w - p_w)
    input_seq = array(raw_seq[ran_ix : ran_ix + w])
    target_seq = array(raw_seq[ran_ix + w : ran_ix + w + p_w])
    input_seq = input_seq.reshape((1, w, n_features))
    
    # Predict the next time stampes of the sampled sequence
    yhat = model.predict(input_seq, verbose=1)
    
    # Print our model's predictions.
    print(yhat)
    
    # Check our predictions against the ground truths.
    print(target_seq) 


.. parsed-literal::

    1/1 [==============================] - 0s 2ms/step
    [[0.10991982]]
    [0.23159759]


.. code:: ipython3

    """Predicting the whole sequence (DeepAnT)"""
    # Build model 
    model = Sequential()
    model.add(Conv1D(filters=num_filt_1,
                     kernel_size=kernel_size,
                     strides=conv_strides,
                     padding='valid',
                     activation='relu',
                     input_shape=(w, n_features)))
    model.add(MaxPooling1D(pool_size=pool_size_1)) 
    model.add(Conv1D(filters=num_filt_2,
                     kernel_size=kernel_size,
                     strides=conv_strides,
                     padding='valid',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size_2))
    model.add(Flatten())
    model.add(Dense(units=num_nrn_dl, activation='relu')) 
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=num_nrn_ol))
    
    # Load the model's saved weights.
    model.load_weights('aws_DeepAnT.h5')
              
    
    input_seq = batch_sample.copy()
    target_seq = batch_label.copy()
    predicted_seq = []
    for i in range(len(input_seq)):
        input = input_seq[i]
        input = input.reshape((1, w, n_features))
        t = model.predict(input, verbose=0)[0][0]
        predicted_seq.append(t)
    # Predict the next time stampes of the sampled sequence
    
    
    # Print our model's predictions.
    # print(predicted_seq)
    
    # Check our predictions against the ground truths.
    # print(target_seq)

.. code:: ipython3

    """Visualization of predicted time series"""
    
    # Acquire the target sequence.
    tar_seq = yahoo['value'][w:]
    # Build up the predicted sequence in dataframe.
    d = {'timestamp': yahoo['timestamp'][w:], 'values': predicted_seq}
    yahoo_pre = pd.DataFrame(data=d)
    pre_seq = yahoo_pre['values']
    
    # plt.plot(in_seq)
    plt.plot(tar_seq)
    plt.plot(pre_seq)
    plt.title('yahoo prediction')
    plt.ylabel('value')
    plt.xlabel('timestamp')
    plt.legend(['target_seq', 'predicted_seq'], loc='upper left')
    axes = plt.gca()
    
    # set the x axis, from w to last index of predicted seq
    axes.set_xlim([w,len(predicted_seq)])
    fig_predict = plt.figure(figsize=(100,10))
    # fig_predict.savefig('predicted_sequence.png')
    plt.show()



.. image:: DeepAnT_1D_yahoo_files/DeepAnT_1D_yahoo_15_0.png



.. parsed-literal::

    <Figure size 7200x720 with 0 Axes>


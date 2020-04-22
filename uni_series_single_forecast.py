""" UNIVARIATE TIME SERIES SINGLE POINT FORECAST
    --------------------------------------------
    Implementation of a lstm recurrent neural network for univariate time series forecasting of a single point in the future.

    This script uses a weather time series dataset, which contains 14 features collected every 10 minutes between 2009 and 2016.

    Code reference: https://www.tensorflow.org/tutorials/structured_data/time_series#single_step_model
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# Set the seed

tf.random.set_seed(13)


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

climate = pd.read_csv('climate dataset/extracted/jena_climate_2009_2016.csv')

print('-'*96, '\n', climate.head())


# Select one feature

uni_data = climate['T (degC)']
uni_data.index = climate['Date Time']

print('-'*96, '\n', uni_data.head())

uni_data = uni_data.values


# Standardize the features using the parameters of the training set

train_split = 300000
uni_train_mean = uni_data[:train_split].mean()
uni_ds_train_std = uni_data[:train_split].std()
uni_data_std = (uni_data - uni_train_mean) / uni_ds_train_std


# Create time series of features and targets for train and valid subsets

def uni_dataset_generator(dataset, start_index, end_index, history_len, target_len):

    end_index = len(dataset) - target_len if end_index is None else end_index

    data = []
    labels = []

    for i in range(start_index, end_index - history_len):

        # range(start_index, start_index + history_len) | range(end_index - 1 - history_len, end_index - 1)
        history_index = range(i, i + history_len)
        data.append(np.reshape(dataset[history_index], (history_len, 1)))

        # (start_index + history_len + target_len) | (end_index - 1 + target_len)
        labels.append(dataset[i + history_len + target_len])

    return np.array(data), np.array(labels)


history_len = 20
target_len = 0

uni_x_train_std, uni_y_train_std = uni_dataset_generator(uni_data_std, 0, train_split, history_len, target_len)
uni_x_valid_std, uni_y_valid_std = uni_dataset_generator(uni_data_std, train_split, None, history_len, target_len)


# Create the train and valid subsets containing tuples of size (20x1) and (1,);
# then cache and shuffle the dataset of tuples and create batches with 256 tuples each

batch_size = 256

uni_ds_train_std = tf.data.Dataset.from_tensor_slices((uni_x_train_std, uni_y_train_std))
uni_ds_train_std = uni_ds_train_std.cache().shuffle(10000).batch(batch_size).repeat()

uni_ds_valid_std = tf.data.Dataset.from_tensor_slices((uni_x_valid_std, uni_y_valid_std))
uni_ds_valid_std = uni_ds_valid_std.batch(batch_size).repeat()

for batch in uni_ds_train_std.take(1):

    array_time_series_of_feature = batch[0]
    array_of_targets = batch[1]

    print('-'*96,
          '\nThe dataset is made up of several batches, each containing an array of 256 time series of the feature'
          '\nand an array of 256 targets. In particular, for each tuple (time series of the feature - target) the',
          '\ntarget is one element after the last element of the time series of the feature.\n',
          '\n*** BATCH 0',
          '\n -- Tuple 0\n', array_time_series_of_feature.numpy()[0], array_of_targets.numpy()[0],
          '\n -- Tuple 1\n', array_time_series_of_feature.numpy()[1], array_of_targets.numpy()[1],
          '\n -- Tuple 254\n', array_time_series_of_feature.numpy()[254], array_of_targets.numpy()[254],
          '\n -- Tuple 255\n', array_time_series_of_feature.numpy()[255], array_of_targets.numpy()[255])


# -------------------------------------------------------------------------------
# 2. DESIGN THE MODEL
# -------------------------------------------------------------------------------


# Design the lstm recurrent neural network

lstm_model = tf.keras.models.Sequential()
lstm_model.add(tf.keras.layers.LSTM(units=8, return_sequences=False, input_shape=uni_x_train_std.shape[-2:]))
lstm_model.add(tf.keras.layers.Dense(1))

""" ALTERNATIVE IMPLEMENTATION:
    --------------------------
    lstm_model.add(tf.keras.layers.RNN(tf.keras.layers.LSTMCell(units=8, input_shape=uni_x_train_std.shape[-2:])))
    
    This requires to delay the "Print the model summary" step until the model has been trained
"""

print('-'*96,
      '\nLSTM LAYER',
      '\nWeight kernel:            4(FE x HU)          = (FE x 4HU) =', lstm_model.weights[0].shape,
      '\nWeight recurrent kernel:  4(HU x HU)          = (HU x 4HU) =', lstm_model.weights[1].shape,
      '\nWeight bias:              4(BS x HU) = 4(HU,) = (4HU,)     =', lstm_model.weights[2].shape,
      '\n-----------------------                                      -------',
      '\nTotal                                                          320')

print('\nDENSE LAYER',
      '\nWeight kernel:                                              ', lstm_model.weights[3].shape,
      '\nWeight :                                                    ', lstm_model.weights[4].shape,
      '\n-------------                                                -------',
      '\nTotal                                                           9')


# Print the model summary

print('-'*96)

lstm_model.summary()

print('Input shape: (time steps x num features) =', uni_x_train_std.shape[-2:],
      '\nNote that the batch size is not specified in "input shape"',
      '\nNote that the number of batches is irrelevant')


# Compile the model to specify optimizer and loss function

lstm_model.compile(optimizer='adam', loss='mae')


# -------------------------------------------------------------------------------
# 3. TRAIN THE MODEL
# -------------------------------------------------------------------------------


# Train the lstm recurrent neural network

print('-' * 96, '\nInput for training: dataset made up of several batches each containing 256 tuples.')

history = lstm_model.fit(uni_ds_train_std, epochs=10, steps_per_epoch=200, validation_data=uni_ds_valid_std, validation_steps=50)


# Visualize the learning curve

hist = history.history

plt.figure()
plt.plot(hist['loss'], 'b', label='Training loss')
plt.plot(hist['val_loss'], 'r', label='Validation loss')
plt.xlabel('Epoch')
plt.title('Training and validation loss')
plt.legend()
plt.tick_params(axis='both', which='major')


# -------------------------------------------------------------------------------
# 3. MAKE PREDICTIONS
# -------------------------------------------------------------------------------


# Create a function to plot the history, true value and model prediction

def plot_prediction(data, delta, title):

    plt.figure()
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'bx', 'rx']

    time_steps = list(range(-data[0].shape[0], 0))

    for i, x in enumerate(data):

        if i:

            plt.plot(delta, data[i], marker[i], label=labels[i])

        else:

            plt.plot(time_steps, data[i].flatten(), marker[i], label=labels[i])

    plt.xlim([time_steps[0], (delta+5)*2])
    plt.xlabel('Time-Step')
    plt.title(title)
    plt.legend()

    return plt


# Make a few predictions

print('-' * 96, '\nInput for predicting: dataset made up of several batches each containing 256 tuples.')

for batch in uni_ds_valid_std.take(3):

    array_time_series_of_feature = batch[0]
    array_of_targets = batch[1]

    prediction = lstm_model.predict(array_time_series_of_feature)

    plot = plot_prediction([array_time_series_of_feature.numpy()[0], array_of_targets.numpy()[0], prediction[0]], 0, 'Simple LSTM model')


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()
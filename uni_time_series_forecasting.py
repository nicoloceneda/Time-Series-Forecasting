""" UNIVARIATE TIME SERIES FORECASTING
    ----------------------------------
    Implementation of a lstm recurrent neural network for univariate time series forecasting.

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

tf.random.set_seed(1)


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

climate = pd.read_csv('climate dataset/extracted/jena_climate_2009_2016.csv')
print('-'*96, '\n', climate.head())


# Select only one feature

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

        # (start_index + history_size + target_size) | (end_index - 1 + target_size)
        labels.append(dataset[i + history_len + target_len])

    return np.array(data), np.array(labels)


history_len = 20
target_len = 0

uni_x_train_std, uni_y_train_std = uni_dataset_generator(uni_data_std, 0, train_split, history_len, target_len)
uni_x_valid_std, uni_y_valid_std = uni_dataset_generator(uni_data_std, train_split, None, history_len, target_len)


# Create the train and valid subsets containing tuples of size (20x1) and (1,1);
# then cache and shuffle the dataset of tuples tuples and create batches with 256 tuples each

batch_size = 256

uni_ds_train_std = tf.data.Dataset.from_tensor_slices((uni_x_train_std, uni_y_train_std))
uni_ds_train_std = uni_ds_train_std.cache().shuffle(10000).batch(batch_size).repeat()

uni_ds_valid_std = tf.data.Dataset.from_tensor_slices((uni_x_valid_std, uni_y_valid_std))
uni_ds_valid_std = uni_ds_valid_std.batch(batch_size).repeat()

for i in uni_ds_train_std.take(1):

    print('-'*96,
          '\nThe dataset is made up of several batches, each containing 256 tuples. In particular, for each',
          '\ntuple, the label is one element after the last element of the time series of the single feature.\n',
          '\n*** BATCH 0 ***',
          '\n -- Tuple 0 --\n', i[0].numpy()[0], i[1].numpy()[0],
          '\n -- Tuple 1 --\n', i[0].numpy()[1], i[1].numpy()[1],
          '\n -- Tuple 254 --\n', i[0].numpy()[254], i[1].numpy()[254],
          '\n -- Tuple 255 --\n', i[0].numpy()[255], i[1].numpy()[255])


# -------------------------------------------------------------------------------
# 2. DESIGN THE MODEL
# -------------------------------------------------------------------------------


# Design the lstm recurrent neural network

simple_lstm_model = tf.keras.models.Sequential()
simple_lstm_model.add(tf.keras.layers.LSTM(units=8, input_shape=uni_x_train_std.shape[-2:]))
simple_lstm_model.add(tf.keras.layers.Dense(1))


# Print the model summary

print('-'*96)
simple_lstm_model.summary()
print('Input shape: (time steps x num features) =', uni_x_train_std.shape[-2:],
      '\nNote that the batch size is not specified in "input shape"',
      '\nNote that the number of batches is irrelevant')


# Compile the model to specify optimizer, loss function

simple_lstm_model.compile(optimizer='adam', loss='mae')


# Make a sample prediction to check the output of the model

for x, y in uni_ds_valid_std.take(1):

    print('Prediction shape:', simple_lstm_model.predict(x).shape, '\n')


# Create a function to plot the history, true value and model prediction

def plot_prediction(plot_data, delta, title):

    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'bx', 'rx']

    delta = delta if delta else 0
    time_steps = list(range(-plot_data[0].shape[0], 0))

    for i, x in enumerate(plot_data):

        if i:

            plt.plot(delta, plot_data[i], marker[i], label=labels[i])

        else:

            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])

    plt.xlim([time_steps[0], (delta+5)*2])
    plt.xlabel('Time-Step')
    plt.title(title)
    plt.legend()

    return plt


# -------------------------------------------------------------------------------
# 3. TRAIN THE MODEL
# -------------------------------------------------------------------------------


# Train the lstm recurrent neural network

history = simple_lstm_model.fit(uni_ds_train_std, epochs=10, steps_per_epoch=200, validation_data=uni_ds_valid_std, validation_steps=50)


# Now that you have trained your simple LSTM, let's try and make a few predictions

for x, y in uni_ds_valid_std.take(3):

    prediction = simple_lstm_model.predict(x)

    plot = plot_prediction([x[0].numpy(), y[0].numpy(), prediction[0]], 0, 'Simple LSTM model')
    plot.show()


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()
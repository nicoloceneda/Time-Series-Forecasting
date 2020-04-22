""" MULTIVARIATE TIME SERIES MULTIPLE POINTS FORECAST
    -------------------------------------------------
    Implementation of a lstm recurrent neural network for multivariate time series forecasting of multiple points in the future.

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


# Select three features

multi_data = climate[['p (mbar)', 'T (degC)', 'rho (g/m**3)']]
multi_data.index = climate['Date Time']

print('-'*96, '\n', multi_data.head())

multi_data = multi_data.values


# Standardize the features using the parameters of the training set

train_split = 300000
multi_train_mean = multi_data[:train_split].mean(axis=0)
multi_train_std = multi_data[:train_split].std(axis=0)
multi_data_std = (multi_data - multi_train_mean) / multi_train_std


# Create time series of features and targets for train and valid subsets

def multi_dataset_generator(dataset, target, start_index, end_index, history_len, target_len, step, single_step):

    end_index = len(dataset) - target_len if end_index is None else end_index

    data = []
    labels = []

    for i in range(start_index, end_index - history_len):

        # range(start_index, start_index + history_len) | range(end_index - 1 - history_len, end_index - 1)
        history_index = range(i, i + history_len, step)
        data.append(dataset[history_index])

        if single_step:

            # (start_index + history_len + target_len) | (end_index - 1 + target_len)
            labels.append(target[i + history_len + target_len])

        else:

            # (start_index + history_len: start_index + history_len + target_len) | (end_index - 1: end_index - 1 + target_len)
            labels.append(target[i + history_len: i + history_len + target_len])

    return np.array(data), np.array(labels)


history_len = 720
target_len = 72
step = 6

multi_multi_x_train_std, multi_multi_y_train_std = multi_dataset_generator(multi_data_std, multi_data_std[:, 1], 0, train_split, history_len, target_len, step, single_step=False)
multi_multi_x_valid_std, multi_multi_y_valid_std = multi_dataset_generator(multi_data_std, multi_data_std[:, 1], train_split, None, history_len, target_len, step, single_step=False)


# Create the train and valid subsets containing tuples of size (120x3) and (72,);
# then cache and shuffle the dataset of tuples and create batches with 256 tuples each

batch_size = 256

multi_ds_train_std = tf.data.Dataset.from_tensor_slices((multi_multi_x_train_std, multi_multi_y_train_std))
multi_ds_train_std = multi_ds_train_std.cache().shuffle(10000).batch(batch_size).repeat()

multi_ds_valid_std = tf.data.Dataset.from_tensor_slices((multi_multi_x_valid_std, multi_multi_y_valid_std))
multi_ds_valid_std = multi_ds_valid_std.batch(batch_size).repeat()

for batch in multi_ds_train_std.take(1):

    array_time_series_of_features = batch[0]
    array_time_series_of_targets = batch[1]

    print('-'*96,
          '\nThe dataset is made up of several batches, each containing an array of 256 time series of the features and an'
          '\narray of 256 time series of the targets. In particular, for each tuple (time series of the features - targets)',
          '\nthe targets are a series of elements 72 elements after the last element of the time series of the features.\n',
          '\n*** BATCH 0',
          '\n -- Tuple 0\n', array_time_series_of_features.numpy()[0], '\n', array_time_series_of_targets.numpy()[0],
          '\n -- Tuple 255\n', array_time_series_of_features.numpy()[255], '\n', array_time_series_of_targets.numpy()[255])


# -------------------------------------------------------------------------------
# 2. DESIGN THE MODEL
# -------------------------------------------------------------------------------


# Design the lstm recurrent neural network

lstm_model = tf.keras.models.Sequential()
lstm_model.add(tf.keras.layers.LSTM(units=32, return_sequences=True, input_shape=multi_multi_x_train_std.shape[-2:]))
lstm_model.add(tf.keras.layers.LSTM(units=16, activation='relu'))
lstm_model.add(tf.keras.layers.Dense(72))


# Print the model summary

print('-'*96)

lstm_model.summary()

print('Input shape: (time steps x num features) =', multi_multi_x_train_std.shape[-2:],
      '\nNote that the batch size is not specified in "input shape"',
      '\nNote that the number of batches is irrelevant')


# Compile the model to specify optimizer and loss function

lstm_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')


# -------------------------------------------------------------------------------
# 3. TRAIN THE MODEL
# -------------------------------------------------------------------------------


# Train the lstm recurrent neural network

print('-' * 96, '\nInput for training: dataset made up of several batches each containing 256 tuples.')

history = lstm_model.fit(multi_ds_train_std, epochs=10, steps_per_epoch=200, validation_data=multi_ds_valid_std, validation_steps=50)


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

def plot_predictions(history, true_future, prediction, title):

    plt.figure()

    num_in = list(range(-len(history), 0))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history), label='History')
    plt.plot(np.arange(num_out)/step, np.array(true_future), 'bo', label='True Future')

    if prediction.any():

        plt.plot(np.arange(num_out)/step, np.array(prediction), 'ro', label='Predicted Future')

    plt.xlabel('Time-Step')
    plt.title(title)
    plt.legend()

    return plt


# Make a few predictions

print('-' * 96, '\nInput for predicting: dataset made up of several batches each containing 256 tuples.')

for batch in multi_ds_valid_std.take(3):

    array_time_series_of_features = batch[0]
    array_time_series_of_targets = batch[1]

    prediction = lstm_model.predict(array_time_series_of_features)

    plot = plot_predictions(array_time_series_of_features.numpy()[0][:, 1], array_time_series_of_targets.numpy()[0], prediction[0], 'Simple LSTM model')


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()
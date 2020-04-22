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


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

climate = pd.read_csv('climate dataset/extracted/jena_climate_2009_2016.csv')
print(climate.head())


# Select only one feature

ds = climate[['p (mbar)', 'T (degC)', 'rho (g/m**3)']]
ds.index = climate['Date Time']
print(ds.head())

ds = ds.values


# Standardize the features using the parameters of the training set

train_split = 300000
ds_train_mean = ds[:train_split].mean(axis=0)
ds_train_std = ds[:train_split].std(axis=0)
ds_std = (ds - ds_train_mean) / ds_train_std


# Generate the train and valid subsets

def dataset_generator(dataset, target, start_index, end_index, history_len, target_len, step, single_step=False):

    end_index = len(dataset) - target_len if end_index is None else end_index

    data = []
    labels = []

    for i in range(start_index, end_index - history_len):

        history_index = range(i, i + history_len, step)  # range(start_index, start_index + history_size) | range(end_index - 1 - history_size, end_index - 1)
        data.append(dataset[history_index])

        if single_step:

            labels.append(target[i + history_len + target_len])  # (start_index + history_size + target_size) | (end_index - 1 + target_size)

        else:

            labels.append(target[i + history_len: i + history_len + target_len])

    return np.array(data), np.array(labels)


history_len = 720
target_len = 72
step = 6

x_train_std_multi, y_train_std_multi = dataset_generator(ds_std, ds_std[:, 1], 0, train_split, history_len, target_len, step, single_step=False)
x_valid_std_multi, y_valid_std_multi = dataset_generator(ds_std, ds_std[:, 1], train_split, None, history_len, target_len, step, single_step=False)


# Cache, shuffle and batch the train and valid subsets

batch_size = 256
tf.random.set_seed(1)

ds_train = tf.data.Dataset.from_tensor_slices((x_train_std_multi, y_train_std_multi))
ds_train = ds_train.cache().shuffle(10000).batch(batch_size).repeat()

ds_valid = tf.data.Dataset.from_tensor_slices((x_valid_std_multi, y_valid_std_multi))
ds_valid = ds_valid.batch(batch_size).repeat()


# -------------------------------------------------------------------------------
# 2. DESIGN THE MODEL
# -------------------------------------------------------------------------------


# Design the lstm recurrent neural network

multi_lstm_step_model = tf.keras.models.Sequential()
multi_lstm_step_model.add(tf.keras.layers.LSTM(units=32, return_sequences=True, input_shape=x_train_std_multi.shape[-2:]))
multi_lstm_step_model.add(tf.keras.layers.LSTM(units=16, activation='relu'))
multi_lstm_step_model.add(tf.keras.layers.Dense(72))


# Print the model summary

multi_lstm_step_model.summary()
print('Input shape:', x_train_std_multi.shape[-2:])


# Compile the model to specify optimizer, loss function

multi_lstm_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')


# Make a sample prediction to check the output of the model

for x, y in ds_valid.take(1):

    print('Prediction shape:', multi_lstm_step_model.predict(x).shape, '\n')


# Train the lstm recurrent neural network

multi_step_history = multi_lstm_step_model.fit(ds_train, epochs=10, steps_per_epoch=200, validation_data=ds_valid, validation_steps=50)


# Plot the training history

plot_train_history(multi_step_history, 'Multi Step Training and validation loss')


# Create a function to plot the history, true value and model prediction

def multi_step_plot(history, true_future, prediction):

  plt.figure(figsize=(12, 6))

  num_in = list(range(-len(history), 0))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/step, np.array(true_future), 'bo', label='True Future')

  if prediction.any():

    plt.plot(np.arange(num_out)/step, np.array(prediction), 'ro', label='Predicted Future')

  plt.legend(loc='upper left')

  return plt


# Now that you have trained your simple LSTM, let's try and make a few predictions

for x, y in ds_valid.take(1):

    prediction = multi_lstm_step_model.predict(x)

    plot = multi_step_plot(x[0], y[0], np.array([0]))
    plot.show()


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()
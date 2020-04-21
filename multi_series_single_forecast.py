""" MULTIVARIATE TIME SERIES FORECASTING
    ------------------------------------
    Implementation of a lstm recurrent neural network for multivariate time series forecasting of a single point in the future.

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

x_train_std_single, y_train_std_single = dataset_generator(ds_std, ds_std[:, 1], 0, train_split, history_len, target_len, step, single_step=True)
x_valid_std_single, y_valid_std_single = dataset_generator(ds_std, ds_std[:, 1], train_split, None, history_len, target_len, step, single_step=True)


# Cache, shuffle and batch the train and valid subsets

batch_size = 256
tf.random.set_seed(1)

ds_train = tf.data.Dataset.from_tensor_slices((x_train_std_single, y_train_std_single))
ds_train = ds_train.cache().shuffle(10000).batch(batch_size).repeat()

ds_valid = tf.data.Dataset.from_tensor_slices((x_valid_std_single, y_valid_std_single))
ds_valid = ds_valid.batch(batch_size).repeat()


# -------------------------------------------------------------------------------
# 2. DESIGN THE MODEL
# -------------------------------------------------------------------------------


# Design the lstm recurrent neural network

simple_lstm_step_model = tf.keras.models.Sequential()
simple_lstm_step_model.add(tf.keras.layers.LSTM(units=32, input_shape=x_train_std_single.shape[-2:]))
simple_lstm_step_model.add(tf.keras.layers.Dense(1))


# Print the model summary

simple_lstm_step_model.summary()
print('Input shape:', x_train_std_single.shape[-2:])


# Compile the model to specify optimizer, loss function

simple_lstm_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')


# Make a sample prediction to check the output of the model

for x, y in ds_valid.take(1):

    print('Prediction shape:', simple_lstm_step_model.predict(x).shape, '\n')


# -------------------------------------------------------------------------------
# 3. TRAIN THE MODEL
# -------------------------------------------------------------------------------


# Train the lstm recurrent neural network

single_step_history = simple_lstm_step_model.fit(ds_train, epochs=10, steps_per_epoch=200, validation_data=ds_valid, validation_steps=50)


# Plot the training history

def plot_train_history(history, title):

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


plot_train_history(single_step_history, 'Single Step Training and validation loss')


# Create a function to plot the history, true value and model prediction

def show_plot(plot_data, delta, title):

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


# Now that you have trained your simple LSTM, let's try and make a few predictions

for x, y in ds_valid.take(3):

    prediction = simple_lstm_step_model.predict(x)

    plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(), prediction[0]], 12, 'Simple Step Prediction')
    plot.show()


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()
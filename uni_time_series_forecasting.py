""" UNIVARIATE TIME SERIES FORECASTING
    ----------------------------------
    Implementation of a lstm recurrent neural network for univariate time series forecasting.
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

ds = climate['T (degC)']
ds.index = climate['Date Time']
print(ds.head())

ds = ds.values


# Standardize the features using the parameters of the training set

train_split = 300000
ds_train_mean = ds[:train_split].mean()
ds_train_std = ds[:train_split].std()
ds_std = (ds - ds_train_mean) / ds_train_std


# Generate the train and valid subsets

def dataset_generator(dataset, start_index, end_index, history_len, target_len):

    end_index = len(dataset) - target_len if end_index is None else end_index

    data = []
    labels = []

    for i in range(start_index, end_index - history_len):

        history_index = range(i, i + history_len)  # range(start_index, start_index + history_size) | range(end_index - 1 - history_size, end_index - 1)
        data.append(np.reshape(dataset[history_index], (history_len, 1)))
        labels.append(dataset[i + history_len + target_len])  # (start_index + history_size + target_size) | (end_index - 1 + target_size)

    return np.array(data), np.array(labels)


history_len = 20
target_len = 0

x_train_std, y_train_std = dataset_generator(ds_std, 0, train_split, history_len, target_len)
x_valid_std, y_valid_std = dataset_generator(ds_std, train_split, None, history_len, target_len)


# Cache, shuffle and batch the train and valid subsets

batch_size = 256
tf.random.set_seed(1)

ds_train = tf.data.Dataset.from_tensor_slices((x_train_std, y_train_std))
ds_train = ds_train.cache().shuffle(10000).batch(batch_size).repeat()

ds_valid = tf.data.Dataset.from_tensor_slices((x_valid_std, y_valid_std))
ds_valid = ds_valid.batch(batch_size).repeat()


# -------------------------------------------------------------------------------
# 2. DESIGN THE MODEL
# -------------------------------------------------------------------------------


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


# Design a baseline model which predicts the next point as the average of the last 20 observations

def baseline(history):

    return np.mean(history)


plot = show_plot([x_train_std[0], y_train_std[0], baseline(x_train_std[0])], 0, 'Baseline Prediction Example')
plot.show()


# Design the lstm recurrent neural network

simple_lstm_model = tf.keras.models.Sequential()
simple_lstm_model.add(tf.keras.layers.LSTM(units=8, input_shape=x_train_std.shape[-2:]))
simple_lstm_model.add(tf.keras.layers.Dense(1))


# Print the model summary

simple_lstm_model.summary()
print('Input shape:', x_train_std.shape[-2:])


# Compile the model to specify optimizer, loss function

simple_lstm_model.compile(optimizer='adam', loss='mae')


# Make a sample prediction to check the output of the model

for x, y in ds_valid.take(1):

    print('Prediction shape:', simple_lstm_model.predict(x).shape, '\n')


# -------------------------------------------------------------------------------
# 3. TRAIN THE MODEL
# -------------------------------------------------------------------------------


# Train the lstm recurrent neural network

history = simple_lstm_model.fit(ds_train, epochs=10, steps_per_epoch=200, validation_data=ds_valid, validation_steps=50)


# Now that you have trained your simple LSTM, let's try and make a few predictions

for x, y in ds_valid.take(3):

    prediction = simple_lstm_model.predict(x)

    plot = show_plot([x[0].numpy(), y[0].numpy(), prediction[0]], 0, 'Simple LSTM model')
    plot.show()


# -------------------------------------------------------------------------------
# 5. GENERAL
# -------------------------------------------------------------------------------


# Show plots

plt.show()
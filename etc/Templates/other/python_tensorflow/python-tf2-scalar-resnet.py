#!/usr/bin/python3

"""

Instructive Example of How Neural Networks Train

"""

import os
import sys
import numba
import numpy as np
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from collections import deque

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM
from tensorflow.keras import losses as KLO
from tensorflow.keras import optimizers as KO
from tensorflow.keras import activations as KA
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Input, Dense, Dropout

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler

sns.set()


def lr_schedule(epoch):
    """
        Learning Rate Schedule.

        Called automatically every epoch as part of callbacks during training.
    """
    lr_max = 0.1
    lr_min = 1e-7
    lr = max(lr_min, lr_max*((1 - 0.02)**epoch))
    return lr


def leaky_relu(x):
    return tf.keras.activations.relu(x, alpha = 0.2)

def ground_truth(x):
    return 20*np.cos(2*np.pi*x/30) + 10*np.sin(2*np.pi*x/60) + (0.04)*x**2 - 30

def ground_truth_cubic(x):
    return 60*np.cos(2*np.pi*x/30) + 40*np.sin(2*np.pi*x/60) + (0.01)*x**2 + (0.0002)*x**3 + 20

x_min = -50
x_max = +50
num_datapoints = 1000
xs_train = np.random.uniform(low = x_min, high = x_max, size = num_datapoints)

# first, try with no noise
# ys_train = ground_truth(xs_train)

# second, try with additive noise
#noise = np.random.normal(loc = 0, scale = 1.00, size = len(xs_train))
#ys_train = noise + ground_truth(xs_train)

# third, try with multiplicative noise
mul_noise = np.random.normal(loc = 1, scale = 0.20, size = len(xs_train))
add_noise = np.random.normal(loc = 0, scale = 5.00, size = len(xs_train))
ys_train = mul_noise * ground_truth(xs_train) + add_noise


lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(
    monitor = 'loss',
    factor = np.sqrt(0.1),
    cooldown = 0,
    patience = 5,
    min_lr = 1e-7
)

callbacks = [lr_reducer, lr_scheduler]



def neuron(x):
    mean = K.mean(x, axis = -1, keepdims = True)
    return tf.keras.activations.relu(x) - mean


input_layer = Input(shape = (1,))
hidden_layer = input_layer

num_blocks = 31
num_layers_per_block = 3
nodes_per_layer = 16

for n in range(num_blocks):

    residuals = hidden_layer

    for _ in range(num_layers_per_block):

        residuals = Dense(
            nodes_per_layer,
            activation = leaky_relu,
        )(residuals)

    residuals = tf.keras.layers.BatchNormalization()(residuals)

    hidden_layer = tf.keras.layers.add([hidden_layer, residuals])

    print(f"Added residual block {n:03d}")


optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule(0))

output_layer = Dense(1, activation = None)(hidden_layer)

print(f"Compiling model.")

model = Model(inputs = input_layer, outputs = output_layer)
model.compile(optimizer = optimizer, loss = 'mse', metrics = ['mse'])

print(f"Model compiled.")

dx = (x_max - x_min) / 1000
xs_test = np.arange(x_min, x_max, dx)
ys_test = ground_truth(xs_test)

epoch = 0

DATA_DIR = f'resnet{num_blocks}' if len(sys.argv) == 1 else sys.argv[1]
PLOT_DIR = f"{DATA_DIR}/plots"
SAVE_DIR = f'{DATA_DIR}/checkpoints'
CHECKPOINT_EVERY = 100

os.makedirs(PLOT_DIR, exist_ok = True)
os.makedirs(SAVE_DIR, exist_ok = True)

history_len = 10
history = deque([], maxlen = history_len)


while True:

    fig, ax = plt.subplots(nrows = 1)
    fig.set_figwidth(2*fig.get_figwidth())
    prediction_axes = ax

    model.fit(xs_train, ys_train, epochs = 1, batch_size = 100, callbacks = callbacks)

    ys_pred = model.predict(xs_test)

    if epoch % 10:
        history.appendleft(ys_pred)

    prediction_axes.set_ylim(-75, +75)
    prediction_axes.plot(xs_train, ys_train, 'o', alpha = 0.2, c = 'g')
    prediction_axes.plot(xs_test, ys_test, alpha = 0.5, c = 'b')

    for n, ys_pred_example in enumerate(history):
        alpha = 1 - (1/history_len)*n
        prediction_axes.plot(xs_test, ys_pred_example, alpha = alpha, c = 'r')

    filename = f"{PLOT_DIR}/{epoch:05d}.png"
    plt.savefig(filename)
    plt.close('all')
    print(f"Wrote {filename}")

    if epoch and (epoch % CHECKPOINT_EVERY == 0):
        save_path = f"{SAVE_DIR}/saved-model-epoch-{epoch:05d}.h5"
        model.save(save_path)

    epoch += 1

# Make a video!
# ffmpeg -f image2 -r 24 -i tf2/%05d.png -vcodec libx264 -y network-training.mp4


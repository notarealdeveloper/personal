#!/usr/bin/python3

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

dataset = tf.keras.datasets.mnist
dataset = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = dataset.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

input_shape = x_train[0].shape


def leaky_relu(x, alpha = 0.2):
    noise_a = np.random.normal(loc = 1, scale = 0.2)
    noise_b = np.random.normal(loc = 1, scale = 0.2)
    return noise_a * keras.activations.relu(x, alpha = alpha * noise_b)


def conv_layer(filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = leaky_relu, dilation_rate = (1, 1)):
    return layers.Conv2D(
        filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        dilation_rate = dilation_rate
    )


def pool_layer(pool_size = (2, 2), padding = 'same'):
    return layers.MaxPool2D(
        pool_size = pool_size,
        padding = padding,
    )

if False:
    model = tf.keras.models.Sequential([
        layers.Flatten(input_shape = input_shape),
        layers.Dense(128, activation = 'relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation = 'softmax')
    ])


model = tf.keras.models.Sequential([

    conv_layer(32),
    layers.Dropout(0.2),

    conv_layer(32),
    layers.Dropout(0.2),

    conv_layer(32),
    layers.Dropout(0.2),

    pool_layer(),

    conv_layer(32),
    layers.Dropout(0.2),

    conv_layer(32),
    layers.Dropout(0.2),

    conv_layer(32),
    layers.Dropout(0.2),

    pool_layer(),

    layers.Flatten(),

    layers.Dense(256, activation = leaky_relu),

    layers.Dropout(0.5),

    layers.Dense(10, activation = 'softmax')
])


model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(x_train, y_train, epochs = 2)

results = {}
eval_results = model.evaluate(x_test, y_test)
for key, value in zip(model.metrics_names, eval_results):
    results[key] = value

print(results)

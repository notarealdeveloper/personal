#!/usr/bin/python3

import functools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.activations as activations

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def plot_data(features, labels):
    xs, ys = features[:,0], features[:,1]
    color = np.where(labels == 0, 'b', 'r')
    df = pd.DataFrame({'x' : xs, 'y' : ys, 'color': color})
    df.plot(
        x = 'x',
        y = 'y',
        kind = 'scatter',
        color = df.color,
        alpha = 0.5,
        title = f"Sample data",
    )
    plt.show()


def make_network(input_shape, output_shape, batch_size = 50):

    input_layer = layers.Input(shape = input_shape, batch_size = batch_size)
    hidden_layer = layers.Dense(32, activation = 'relu')(input_layer)
    hidden_layer = layers.Dense(32, activation = 'relu')(hidden_layer)
    hidden_layer = layers.Dense(32, activation = 'relu')(hidden_layer)
    output_layer = layers.Dense(output_shape, activation = 'softmax')(hidden_layer)

    model = models.Model(inputs = input_layer, outputs = output_layer)
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'],
    )

    return model


def main():

    # collect dataset
    noise = 0.15
    features, labels = make_moons(n_samples = 1000, noise = noise)

    # visualize dataset
    plot_data(features, labels)

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(features, labels)

    # make network
    input_shape = x_train.shape[1:]     # keras shape doesn't count batch size
    output_shape = len(set(y_train))    # n_categories
    model = make_network(input_shape, output_shape)

    # train model
    callbacks = []
    callback = tf.keras.callbacks.ProgbarLogger()
    callback.set_model(model)
    callbacks.append(callback)
    num_epochs = 100

    model.fit(x_train, y_train, epochs = num_epochs, callbacks = callbacks)

    # make predictions
    logits = model.predict(x_test)
    y_pred = np.argmax(logits, axis = 1)

    # evaluate our predictions
    accuracy = (y_test == y_pred).mean()
    print(f"accuracy is {100*accuracy}%")


if __name__ == '__main__':
    main()


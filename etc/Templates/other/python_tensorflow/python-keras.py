#!/usr/bin/env python3

import os
import io
import re
import sys
import glob
import json
import time
import pickle
import requests
import argparse
import contextlib
import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp

import keras
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import BatchNormalization

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data['data'], data['target']
ohe = OneHotEncoder(n_values = 3, sparse = False)
y_oh = ohe.fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y_oh)


model = Sequential()
model.add(Dense(units = 4, activation='relu', input_dim = 4))

for n in range(8):
    model.add(Dense(
        units = 8,
        activation = 'relu',
    ))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())


model.add(Dense(units = 3, activation = 'softmax'))

model.compile(
    loss = keras.losses.categorical_crossentropy,
    optimizer = 'adam',
    metrics = ['accuracy']
)

dev_null = open('/dev/null', 'w')

for n in range(20):

    with contextlib.redirect_stdout(dev_null):
        model.fit(X_train, y_train, epochs = 100, batch_size = 64)

    # evaluate performance on test data
    with contextlib.redirect_stdout(dev_null):
        loss, accuracy = model.evaluate(X_test, y_test, batch_size=len(X_test))

    print(f"accuracy is {accuracy}, loss is {loss}")

    #time.sleep(1)

    # generate predictions on new data
    # classes = model.predict(X_test, batch_size = 128)


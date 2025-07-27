#!/usr/bin/python3 -B

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import GRUBlockCell
from tensorflow.contrib.rnn import LSTMBlockCell
from tensorflow.contrib.rnn import DropoutWrapper

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

from jlib import colors


def initialize_session(model_dir):

    """ Create a session and saver initialized from a checkpoint if found. """

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    model_dir = os.path.realpath(model_dir)

    checkpoint = tf.train.latest_checkpoint(model_dir)

    saver = tf.train.Saver()
    sess = tf.InteractiveSession(config = config)

    if checkpoint:
        saver.restore(sess, checkpoint)
    else:
        os.makedirs(model_dir, exist_ok = True)
        sess.run(tf.global_variables_initializer())
    return (sess, saver)


def get_batch(X, y, batch_size = 500):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.cache().shuffle(buffer_size = 1000).batch(batch_size)
    dataset = dataset.repeat().prefetch(1)
    next_op = dataset.make_one_shot_iterator().get_next()
    return next_op


class Config:

    def __init__(self, model_dir):

        self.model_dir = model_dir
        self.log_dir = f'{self.model_dir}/logs'
        self.log_dir_train = f'{self.log_dir}/train'
        self.log_dir_test  = f'{self.log_dir}/test'
        self.checkpoint_dir = f'{self.model_dir}/checkpoint'

        os.makedirs(self.model_dir, exist_ok = True)
        os.makedirs(self.log_dir, exist_ok = True)
        os.makedirs(self.log_dir_train, exist_ok = True)
        os.makedirs(self.log_dir_test, exist_ok = True)
        os.makedirs(self.checkpoint_dir, exist_ok = True)


with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    data = fetch_mldata('MNIST original')


X, y = data.data, data.target

x_dim, y_dim = 28, 28

X = X.reshape(-1, x_dim, y_dim)
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y)

num_timesteps = x_dim
num_features = y_dim
num_hidden_rnn = 64
num_layers_rnn = 6
num_units_dense = 32
num_classes = 10

current_learning_rate = 0.01
current_dense_dropout_rate = 0.5
current_rnn_dropout_rate = 0.5


with tf.name_scope('input'):

    inputs = tf.placeholder(tf.float32, [None, x_dim, y_dim], name = 'inputs')
    image_cols = tf.identity(inputs, name = 'image_cols')
    image_rows = tf.transpose(inputs, [0, 2, 1], name = 'image_rows')
    image_data = tf.concat([image_cols, image_rows], 1, name = 'image_data')

    learning_rate = tf.placeholder_with_default(1e-4, [])
    dense_dropout_rate = tf.placeholder_with_default(0.0, [])
    rnn_dropout_rate = tf.placeholder_with_default(0.0, [])

    rnn_keep_prob = 1 - rnn_dropout_rate


with tf.name_scope('labels'):
    labels = tf.placeholder(tf.int64, [None])
    onehot_labels = tf.one_hot(labels, 10)


with tf.name_scope('rnn'):

    raw_cells = [LSTMBlockCell(num_hidden_rnn) for _ in range(num_layers_rnn)]

    rnn_cells = [
        DropoutWrapper(cell, input_keep_prob = rnn_keep_prob, output_keep_prob = rnn_keep_prob)
        for cell in raw_cells
    ]

    cell = MultiRNNCell(rnn_cells)
    rnn_hidden, rnn_state = tf.nn.dynamic_rnn(cell, image_data, dtype = tf.float32)
    rnn_output = rnn_hidden[:, -1, :]


with tf.name_scope('dense'):
    dense1 = tf.layers.dense(rnn_output, num_units_dense)
    dropout1 = tf.layers.dropout(dense1, rate = dense_dropout_rate)
    logits = tf.layers.dense(dropout1, num_classes)

with tf.name_scope('loss'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)

with tf.name_scope('optimize'):
    optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.name_scope('metrics'):
    prediction = tf.argmax(logits, 1, name = 'prediction')
    equality = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))


tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('loss', loss)

tf.summary.scalar('learning_rate', learning_rate)
tf.summary.scalar('dense_dropout_rate', dense_dropout_rate)
tf.summary.scalar('rnn_dropout_rate', rnn_dropout_rate)

tf.summary.histogram('accuracy', accuracy)
tf.summary.histogram('loss', loss)


### Session Setup
global_step = tf.train.get_or_create_global_step()
step = tf.assign_add(global_step, 1)

merged = tf.summary.merge_all()

get_train_batch = get_batch(X_train, y_train, batch_size = 1000)
get_test_batch  = get_batch(X_test, y_test, batch_size = len(X_test))

config = Config('model')
sess, saver = initialize_session(config.checkpoint_dir)
train_writer = tf.summary.FileWriter(config.log_dir_train, sess.graph)
test_writer = tf.summary.FileWriter(config.log_dir_test, sess.graph)


while True:

    X_train_batch, y_train_batch = sess.run(get_train_batch)

    epoch, loss_val, optimize_val, accuracy_val, summary = sess.run(
        [step, loss, optimize, accuracy, merged],
        feed_dict = {
            inputs: X_train_batch,
            labels: y_train_batch,
            learning_rate: current_learning_rate,
            dense_dropout_rate: current_dense_dropout_rate,
            rnn_dropout_rate: current_rnn_dropout_rate,
        }
    )

    train_writer.add_summary(summary, epoch)

    if epoch % 20 == 0:
        print(f"{colors.blue}TRAIN: {colors.white}epoch {epoch}: "
              f"loss is {loss_val:.08f}, accuracy is {accuracy_val:.08f}{colors.end}")

        #print(rnn_output_val)
        #print(rnn_output_val.shape)

    if epoch % 100 == 0:

        X_test_batch, y_test_batch = sess.run(get_test_batch)

        loss_val, accuracy_val, summary = sess.run(
            [loss, accuracy, merged],
            feed_dict = {
                inputs: X_test_batch,
                labels: y_test_batch,
            }
        )

        test_writer.add_summary(summary, epoch)

        print(f"{colors.red}TEST!: {colors.green}epoch {epoch}: "
              f"loss is {loss_val:.08f}, accuracy is {accuracy_val:.08f}{colors.end}")

    # adjust our hyperparameters throughout training
    if (epoch % 500 == 0) and current_learning_rate >= 1e-4:
        current_learning_rate *= 0.80

    if (epoch % 500 == 0) and current_dense_dropout_rate >= 0.1:
        current_dense_dropout_rate *= 0.95

    if (epoch % 500 == 0) and current_rnn_dropout_rate > 0.1:
        current_rnn_dropout_rate *= 0.95

    if (epoch % 1000 == 0):
        saver.save(sess, f'{config.checkpoint_dir}/model.ckpt', epoch)


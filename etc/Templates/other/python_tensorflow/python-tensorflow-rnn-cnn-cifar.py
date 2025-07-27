#!/usr/bin/python3 -B

import os
import sys
import time
import glob
import pickle
import tarfile
import hashlib
import requests
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
from jlib import tf_initialize_session
from jlib import tf_get_batch
from datasets import Cifar10


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



def get_cifar_10():
    cmd = "locate -r 'cifar-10/cifar-10-batches-py$'"
    cifar_10_paths = os.popen(cmd).read().splitlines()
    if len(cifar_10_paths) > 0:
        path = cifar_10_paths[0]
        cifar = Cifar10(os.path.dirname(path))
        print(f"Found local cifar-10 in {path}. Parsing .")
    else:
        cifar = Cifar10()
    return cifar


cifar = get_cifar_10()
X_train = cifar.X_train
X_test = cifar.X_test
y_test = cifar.y_test
y_train = cifar.y_train


x_dim, y_dim = 32, 32
num_channels = 3
num_classes = 10


with tf.name_scope('inputs'):
    inputs = tf.placeholder(tf.float32, [None, x_dim, y_dim, 3], name = 'X')
    labels = tf.placeholder(tf.int64, [None], name = 'y')
    onehot_labels = tf.one_hot(labels, 10)


with tf.name_scope('globals'):
    learning_rate       = tf.Variable(0.01, name = 'learning_rate')
    dense_dropout_rate  = tf.Variable(0.5, name = 'dense_dropout_rate')
    rnn_dropout_rate    = tf.Variable(0.5, name = 'rnn_dropout_rate')
    rnn_keep_prob       = 1 - rnn_dropout_rate


with tf.name_scope('cnn'):

    conv2d1  = tf.layers.conv2d(inputs, 64, [3, 3], activation = tf.nn.leaky_relu)
    maxpool1 = tf.layers.max_pooling2d(conv2d1, [3, 3], 1, padding = 'same')

    conv2d2  = tf.layers.conv2d(maxpool1, 32, [3, 3], activation = tf.nn.leaky_relu)
    maxpool2 = tf.layers.max_pooling2d(conv2d2, [3, 3], 1, padding = 'same')

    cnn_flat = tf.layers.flatten(maxpool2)

    cnn_output = tf.layers.dense(cnn_flat, 128)

    #print(cnn_output)


with tf.name_scope('rnn'):

    num_hidden_rnn = 32
    num_layers_rnn = 2

    image_cols = tf.identity(inputs, name = 'image_cols')
    image_rows = tf.transpose(inputs, [0, 2, 1, 3], name = 'image_rows')
    image_data = tf.concat([image_cols, image_rows], 1, name = 'image_data')

    image_input_data = tf.concat([image_data[:,:,:,n] for n in range(3)], 2, name = 'image_input_data')

    raw_cells = [LSTMBlockCell(num_hidden_rnn) for _ in range(num_layers_rnn)]

    rnn_cells = [
        DropoutWrapper(cell, input_keep_prob = rnn_keep_prob, output_keep_prob = rnn_keep_prob)
        for cell in raw_cells
    ]

    cell = MultiRNNCell(rnn_cells)
    rnn_hidden, rnn_state = tf.nn.dynamic_rnn(cell, image_input_data, dtype = tf.float32)
    rnn_output = tf.layers.dense(rnn_hidden[:, -1, :], 128)

    #print(rnn_output)


with tf.name_scope('dense'):

    dense_input = tf.concat([cnn_output, rnn_output], 1)

    dense = tf.layers.dense(dense_input, 64)
    dropout = tf.layers.dropout(dense, rate = dense_dropout_rate)

    logits = tf.layers.dense(dropout, num_classes)


with tf.name_scope('loss'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)

with tf.name_scope('optimize'):
    optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.name_scope('metrics'):

    prediction = tf.argmax(logits, 1, name = 'prediction')
    equality = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    tf.summary.histogram('accuracy', accuracy)
    tf.summary.histogram('loss', loss)

    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('loss', loss)


with tf.name_scope('rates'):
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('dense_dropout_rate', dense_dropout_rate)
    tf.summary.scalar('rnn_dropout_rate', rnn_dropout_rate)


### Session Setup
global_step = tf.train.get_or_create_global_step()
step = tf.assign_add(global_step, 1)

merged = tf.summary.merge_all()

get_train_batch = tf_get_batch(X_train, y_train, batch_size = 1000)
get_test_batch  = tf_get_batch(X_test, y_test, batch_size = 1000)

config = Config('model')

train_writer = tf.summary.FileWriter(config.log_dir_train)
test_writer = tf.summary.FileWriter(config.log_dir_test)

sess, saver = tf_initialize_session(config.checkpoint_dir)

def write_frozen_pb_file(config, sess, epoch):

    graph = sess.graph

    graph_def = graph.as_graph_def()

    filename = f"{config.model_dir}/cifar-10-epoch-{epoch:06d}.pb"

    output_names = [
        'inputs/X',
        'inputs/y',
        'globals/learning_rate',
        'globals/rnn_dropout_rate',
        'globals/dense_dropout_rate',
    ]

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,           # the session is used to get the weights
        graph_def,      # the graph_def is used to get the nodes 
        output_names    # the output node names are used to select the useful nodes
    )

    with tf.gfile.GFile(filename, 'wb') as fp:
        fp.write(frozen_graph_def.SerializeToString())


try:

    while True:

        X_train_batch, y_train_batch = sess.run(get_train_batch)

        epoch, loss_val, optimize_val, accuracy_val, summary = sess.run(
            [step, loss, optimize, accuracy, merged],
            feed_dict = {
                inputs: X_train_batch,
                labels: y_train_batch,
            }
        )

        train_writer.add_summary(summary, epoch)

        if epoch % 20 == 0:
            print(f"{colors.blue}TRAIN: {colors.white}epoch {epoch}: "
                  f"loss is {loss_val:.08f}, accuracy is {accuracy_val:.08f}{colors.end}")


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
        if (epoch % 500 == 0) and sess.run(learning_rate) >= 5e-5:
            sess.run(tf.assign(learning_rate, 0.95*learning_rate))
            #sess.run(tf.assign(learning_rate, 0.01))

        if (epoch % 500 == 0) and sess.run(dense_dropout_rate) >= 0.1:
            pass #sess.run(tf.assign(dense_dropout_rate, 0.95*dense_dropout_rate))
            #sess.run(tf.assign(dense_dropout_rate, 0.5))

        if (epoch % 100 == 0) and sess.run(rnn_dropout_rate) > 0.1:
            pass #sess.run(tf.assign(rnn_dropout_rate, 0.95*rnn_dropout_rate))
            #sess.run(tf.assign(rnn_dropout_rate, 0.5))

        if (epoch % 1000 == 0):
            saver.save(sess, f'{config.checkpoint_dir}/model.ckpt', epoch)

        if (epoch % 100_000 == 0):
            write_frozen_pb_file(config, sess, epoch)

except KeyboardInterrupt:
    print(f"\b\bSaving checkpoint.")
    saver.save(sess, f'{config.checkpoint_dir}/model.ckpt', epoch)
    sys.exit(0)


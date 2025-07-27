#!/usr/bin/env python3

import os
import re
import sys
import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

from collections import defaultdict
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor

def initialize_session(checkpoint_dir):

    """ Create a session and saver initialized from a checkpoint if found. """

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    checkpoint_dir = os.path.realpath(checkpoint_dir)
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    saver = tf.train.Saver()

    sess = tf.InteractiveSession(config = config)
    if checkpoint:
        saver.restore(sess, checkpoint)
    else:
        os.makedirs(checkpoint_dir, exist_ok = True)
        sess.run(tf.global_variables_initializer())
    return (sess, saver)


def make_batch(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.cache().shuffle(buffer_size = 1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat().prefetch(1)
    next_op = dataset.make_one_shot_iterator().get_next()
    return next_op


def conv_layer(inputs, channels = 64, kernel_size = (3, 3), padding = 'valid', strides = (1, 1)):
    return tf.layers.conv2d(
        inputs,
        channels,
        kernel_size,
        padding = padding,
        activation = tf.nn.leaky_relu,
        strides = strides
    )


learning_rate = 0.0005
num_epochs = 4000
batch_size = 100

### Data Setup
#(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()


y_train[np.where(y_train == 9)] = 1     # Trucks and cars are the fucking same you idiots
y_test[np.where(y_test == 9)] = 1

#y_train[np.where(y_train == 7)] = 3     # Horses and deer are pretty fucking similar too.
#y_test[np.where(y_test == 7)] = 3
#y_train[np.where(y_train == 8)] = 7     # Set the 8 label to 7 so one-hot doesn't get confused.
#y_test[np.where(y_test == 8)] = 7

### Graph Setup
num_categories = len(np.unique(y_train))
feature_shape = X_train.shape[1:]
labels_shape = y_train.shape[1:]

inputs = tf.placeholder(tf.float32, [None, *feature_shape], name = 'inputs')
labels = tf.placeholder(tf.int32,   [None, *labels_shape],  name = 'labels')

onehot_labels = tf.one_hot(tf.squeeze(labels, axis = 1), num_categories)
dropout_rate = tf.placeholder_with_default(0.0, ())

conv2d0 = inputs
conv2d1 = conv_layer(conv2d0, channels = 64)
conv2d2 = conv_layer(conv2d1, channels = 64)
conv2d3 = conv_layer(conv2d2, channels = 64)
conv2d4 = conv_layer(conv2d3, channels = 64)

flattened = tf.layers.flatten(conv2d4)

#dropout = tf.layers.dropout(flattened, rate = dropout_rate)
#dense = tf.layers.dense(dropout, 64, activation = tf.nn.leaky_relu)
dense = tf.layers.dense(flattened, 64, activation = tf.nn.leaky_relu)
logits = tf.layers.dense(dense, num_categories, activation = None)

loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)

optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)

prediction = tf.argmax(logits, 1, name = 'prediction')
equal_op = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(equal_op, tf.float32))

### Session Setup
global_step = tf.train.get_or_create_global_step()
step = tf.assign_add(global_step, 1)

checkpoint_dir = 'model'

sess, saver = initialize_session(checkpoint_dir)

get_train_batch = make_batch(X_train, y_train, batch_size = batch_size)
get_test_batch  = make_batch(X_test, y_test, batch_size = 2500)


while sess.run(global_step) < num_epochs:

    X_train_batch, y_train_batch = sess.run(get_train_batch)

    epoch, loss_val, optimize_val, accuracy_val = sess.run(
        [step, loss, optimize, accuracy],
        feed_dict = {inputs: X_train_batch, labels: y_train_batch, dropout_rate: 0.5}
    )

    if epoch % 100 == 0:
        X_test_batch, y_test_batch = sess.run(get_test_batch)
        test_accuracy_val = sess.run(accuracy, feed_dict = {inputs: X_test_batch, labels: y_test_batch})
        print(f"{epoch:05d}: accuracy on test set batch is: {100*test_accuracy_val:.02f}")

    if epoch % 1000 == 0:
        saver.save(sess, f'{checkpoint_dir}/model.ckpt', global_step = global_step)


### test the model
accuracy_val = sess.run(accuracy, feed_dict = {inputs: X_test_batch, labels: y_test_batch})
print(f"Accuracy on test set is: {accuracy_val}")


### pb file savers and loaders
def write_frozen_pb_file(filename, sess, graph, output_names):
    graph_def = graph.as_graph_def()

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,           # the session is used to get the weights
        graph_def,      # the graph_def is used to get the nodes 
        output_names    # the output node names are used to select the useful nodes
    )

    with tf.gfile.GFile(filename, 'wb') as fp:
        fp.write(frozen_graph_def.SerializeToString())


def read_frozen_pb_file(filename, prefix = ''):

    # Load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(filename, "rb") as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name = prefix)
    return graph


### write frozen pb file
filename = 'frozen.pb'
graph = tf.get_default_graph()
nodes = ['prediction', 'inputs']
write_frozen_pb_file(filename, sess, graph, nodes) # hooray!


### close the old session
sess.close()


### read frozen pb file
graph = read_frozen_pb_file('frozen.pb')
sess = tf.InteractiveSession(graph = graph)

### examine the just loaded graph
#for op in graph.get_operations():
#    print(op.name)

### load tensors from the pb file

inputs     = graph.get_tensor_by_name('inputs:0')
prediction = graph.get_tensor_by_name('prediction:0')

# Get random element from the test set
index = random.randint(0, len(X_test))
X, [y_true] = X_test[index], y_test[index]

[y_pred] = sess.run(prediction, feed_dict = {inputs: [X]})

print(f"y_pred is {y_pred}, y_true is {y_true}.")

plt.imshow(X); plt.show()




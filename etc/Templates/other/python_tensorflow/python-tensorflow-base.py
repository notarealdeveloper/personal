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

from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

from collections import defaultdict
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor

learning_rate = 0.01
num_epochs = 1000
test_size = 0.2

### Data Setup
digits = fetch_mldata('MNIST original')
#digits = load_digits()
X = digits.data
y = digits.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

### Graph Setup
num_features = X.shape[1]
num_categories = 10 # This can actually be set to higher than 10, and the model still does well.

inputs = tf.placeholder(tf.float32, [None, num_features], name = 'inputs')
labels = tf.placeholder(tf.int32,   [None, 1])

onehot_labels = tf.one_hot(labels, num_categories)
onehot_labels = tf.reshape(onehot_labels, (-1, num_categories))

dense1 = tf.layers.dense(inputs, num_features, activation = tf.nn.relu)
dense2 = tf.layers.dense(dense1, 200, activation = tf.nn.relu)
dense3 = tf.layers.dense(dense2, 100, activation = tf.nn.relu)
dense4 = tf.layers.dense(dense3, 50, activation = tf.nn.relu)
logits = tf.layers.dense(dense4, num_categories, activation = None)

loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=onehot_labels))

#optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss) # much better for this network than GD

prediction = tf.argmax(logits, 1, name = 'prediction')
equal_op = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(equal_op, tf.float32))

### Session Setup
global_step = tf.train.get_or_create_global_step()
step = tf.assign_add(global_step, 1)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)


epoch = 0

while epoch < num_epochs:

    # TODO: Add batching logic here using tf.data

    epoch, loss_val, optimize_val = sess.run(
        [step, loss, optimize],
        feed_dict = {inputs: X_train, labels: y_train}
    )

    print(f"epoch is {epoch}, loss is {loss_val}")


### test the model
accuracy_val = sess.run(accuracy, feed_dict = {inputs: X_test, labels: y_test})
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


### examine the just loaded graph
for op in graph.get_operations():
    print(op.name)

### load tensors from the pb file
prediction = graph.get_tensor_by_name('prediction:0')
inputs = graph.get_tensor_by_name('inputs:0')

### make predictions from the just loaded pb file
times_to_show_off = 100

os.makedirs('example-wins', exist_ok = True)
os.makedirs('example-fails', exist_ok = True)
pool = ProcessPoolExecutor(max_workers = mp.cpu_count())

def plot_example_image(example_image, filename):
    plt.imshow(example_image.reshape(28, 28), cmap = 'Greys', interpolation='nearest')
    plt.savefig(filename)
    return filename

futures = []

with tf.Session(graph = graph) as sess:

    for n in range(len(X_test)):

        example_image = X_test[n].reshape(1, -1)
        correct_value = y_test[n]

        predicted_val = sess.run(
            prediction,
            feed_dict = {
                inputs: example_image,
            }
        )

        y_pred = predicted_val[0]
        y_true = correct_value[0]
        print(f"predicted value is {y_pred}, correct value is {y_true}")

        guessed_correct = (y_pred == y_true)
        example_dir = 'example-wins' if guessed_correct else 'example-fails'
        filename = f'{example_dir}/{n:05d}_pred-{y_pred}_true-{y_true}.png'

        future = pool.submit(plot_example_image, example_image, filename)
        futures.append(future)

for future in as_completed(futures):
    print(f"Finished writing {future.result()}")

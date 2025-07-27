#!/usr/bin/env python3

import os
import sys
import random
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf


class RecurrentSequenceModel:

    def __init__(self, num_layers, num_units, onehot_depth = 256):

        self.onehot_depth = onehot_depth

        self.optimizer = tf.train.AdamOptimizer

        self.cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.GRUBlockCell(num_units) for _ in range(num_layers)]
        )

        self.output_layer = tf.layers.Dense(self.onehot_depth, activation = None)


    def optimize(self, chunks, length, learning_rate):

        chunks = tf.one_hot(chunks, self.onehot_depth)

        inputs, targets = chunks[:, :-1], chunks[:, 1:] # time is the 2nd tensor index

        hidden, state = tf.nn.dynamic_rnn(self.cell, inputs, length, dtype = tf.float32)

        logits = self.output_layer(hidden)

        loss = tf.losses.softmax_cross_entropy(targets, logits)

        optimize = self.optimizer(learning_rate).minimize(loss)

        # force the optimization step to run before the loss is returned
        with tf.control_dependencies([optimize]):
            return tf.identity(loss)


    def generate(self, seed, length, temperature):

        inputs = tf.one_hot(seed[:, :-1], self.onehot_depth)

        sequence_indices = tf.range(length)

        hidden, state = tf.nn.dynamic_rnn(self.cell, inputs, dtype = tf.float32)

        def generate_next(values, unknown):
            token, state = values
            onehot_token = tf.one_hot(token, self.onehot_depth)
            hidden, new_state = self.cell(onehot_token, state)
            logits = self.output_layer(hidden)
            new_token = tf.distributions.Categorical(logits / temperature).sample()
            return (tf.cast(new_token, tf.uint8), new_state)

        tokens, unknowns = tf.scan(generate_next, sequence_indices, (seed[:, -1], state))

        return tf.transpose(tokens, perm = [1, 0])


def initialize_session(args):

    """ Create a session and saver initialized from a checkpoint if found. """

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    model_dir = os.path.realpath(args.model_dir)

    checkpoint = tf.train.latest_checkpoint(model_dir)

    saver = tf.train.Saver()
    sess = tf.InteractiveSession(config = config)

    if checkpoint:
        saver.restore(sess, checkpoint)
    else:
        os.makedirs(model_dir, exist_ok = True)
        sess.run(tf.global_variables_initializer())
    return (sess, saver)


def training(args):

    """Train the model and frequently print the loss and save checkpoints."""

    print("Parsing input data")

    dataset = tf.data.TextLineDataset([args.corpus])

    dataset = dataset.map(lambda line: tf.decode_raw(line, tf.uint8))

    dataset = dataset.flat_map(lambda line: chunk_sequence(line, args.chunk_length))

    dataset = dataset.cache().shuffle(buffer_size = 1000).batch(args.batch_size)

    dataset = dataset.repeat().prefetch(1)

    chunks, length = dataset.make_one_shot_iterator().get_next()

    print("Instantiating model")

    model = RecurrentSequenceModel(args.num_layers, args.num_units)

    loss = model.optimize(chunks, length, args.learning_rate)

    step = tf.train.get_or_create_global_step()
    increment_step = step.assign_add(1)

    print("Initializing session")

    sess, saver = initialize_session(args)

    epoch_digits = int(np.log10(args.total_steps)) + 1
    format_epoch = lambda n: f"{n:0{epoch_digits}d}"

    while True:

        epoch = sess.run(step)

        print(f"Beginning epoch {format_epoch(epoch)}")

        if epoch >= args.total_steps:
            print('Training complete.')
            break

        loss_value, step_value = sess.run([loss, increment_step])

        if step_value % args.log_every == 0:
            print(f'Epoch {step_value}, loss is {loss_value}.')

        if step_value % args.checkpoint_every == 0:
            print('Saving checkpoint.')
            saver.save(sess, os.path.join(args.model_dir, 'model.ckpt'), step_value)

        if step_value % args.speak_every == 0:
            progname = os.path.realpath(os.path.basename(sys.argv[0]))
            os.system(f"{progname} --mode sampling --model-dir {args.model_dir} --corpus {args.corpus}")



def chunk_sequence(sequence, chunk_length):

    """ Split a sequence tensor into a batch of zero-padded chunks. """

    num_chunks = 1 + ((tf.shape(sequence)[0] - 1) // chunk_length)

    padding_length = chunk_length * num_chunks - tf.shape(sequence)[0]

    padding = tf.zeros(
        tf.concat([[padding_length], tf.shape(sequence)[1:]], 0),
        sequence.dtype
    )

    padded = tf.concat([sequence, padding], 0)

    chunks = tf.reshape(padded, [
      num_chunks, chunk_length] + padded.shape[1:].as_list()
    )

    length = tf.concat([
        chunk_length * tf.ones([num_chunks - 1], dtype=tf.int32),
        [chunk_length - padding_length]], 0
    )

    return tf.data.Dataset.from_tensor_slices((chunks, length))



def seed_generator():
    import string
    letters = string.ascii_uppercase
    while True:
        yield ord(random.choice(letters))
        # yield ord(random.choice('JGEK'))

def sampling(args, num_samples = 10):

    """ Start from user provided input sequence, and pontificate forever. """

    model = RecurrentSequenceModel(args.num_layers, args.num_units)
    seed = tf.placeholder(tf.uint8, [None, None])
    temp = tf.placeholder(tf.float32, [])
    text = tf.concat([seed, model.generate(seed, args.sample_length, temp)], 1)

    sess, saver = initialize_session(args)

    g = seed_generator()
    n = 0

    array_from_seed = lambda s: [[ord(c) for c in s]]

    while n < num_samples:
        # seed_value = array_from_seed('KRAMER:')
        seed_value = [[next(g)]]
        temp_value = 1.0
        for numpy_ascii_array in sess.run(text, {seed: seed_value, temp: temp_value}):

            text_value = numpy_ascii_array.tobytes().decode(errors = 'replace')

            print(text_value + '\n')

        n += 1

    sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default = 'training', choices = ['training', 'sampling'])

    parser.add_argument('--model-dir', default = 'model')

    parser.add_argument('--corpus', required = True)

    parser.add_argument('--batch-size', type = int, default = 100)

    parser.add_argument('--chunk-length', type = int, default = 256)

    parser.add_argument('--learning-rate', type = float, default = 1e-4)

    parser.add_argument('--num-units', type = int, default = 512)

    parser.add_argument('--num-layers', type = int, default = 3)

    parser.add_argument('--total-steps', type = int, default = 1_000_000)

    parser.add_argument('--checkpoint-every', type = int, default = 1000)

    parser.add_argument('--log-every', type = int, default = 1000)

    parser.add_argument('--speak-every', type = int, default = 1000)

    parser.add_argument('--sample-length', type = int, default = 500)

    args = parser.parse_args()

    if args.mode == 'training':
        training(args)

    if args.mode == 'sampling':

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        sampling(args)


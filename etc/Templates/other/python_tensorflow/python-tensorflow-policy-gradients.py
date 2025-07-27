#!/usr/bin/env python3

import os
import sys
import gym
import numba
import collections
import numpy as np
import pandas as pd
import tensorflow as tf


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


def autoclose_environment(func):
    def wrapped(env, *args, **kwargs):
        try:
            func(env, *args, **kwargs)
        except Exception as exc:
            env.close()
            raise exc
    return wrapped


@autoclose_environment
def policy_gradients(env):

    """
        Implementing policy gradients.

        How to learn a policy that maps observations to actions.
    """

    n_inputs  = env.observation_space.shape[0]
    n_hidden  = 2*n_inputs # for cartpole, this is just 4.
    n_outputs = 1 # for now we'll just output one logit (pre-probability) and do the multinomial math

    recent_reward_buffer = collections.deque([], maxlen = 20)

    initializer = tf.initializers.variance_scaling()

    inputs = tf.placeholder(tf.float32, [None, n_inputs])
    learning_rate = tf.placeholder(tf.float32, [])

    hidden1 = tf.layers.dense(
        inputs,
        n_hidden,
        activation = tf.nn.leaky_relu,
        kernel_initializer = initializer
    )

    hidden2 = tf.layers.dense(
        hidden1,
        n_hidden,
        activation = tf.nn.leaky_relu,
        kernel_initializer = initializer
    )

    dropout = tf.layers.dropout(hidden2, rate = 0.5)

    logits = tf.layers.dense(
        dropout,
        n_outputs,
        activation = None,
        kernel_initializer = initializer
    )

    outputs = tf.nn.sigmoid(logits)

    probabilities = tf.concat(axis = 1, values = [outputs, 1 - outputs])

    action = tf.multinomial(tf.log(probabilities), num_samples = 1)

    # Assignment of Credit!
    #
    # Here's where things get interesting.

    # We're going to trick the mathematics into thinking of this as a supervised learning problem.
    # (Even though we have no labels, and we won't get any feedback until the end of each episode.)

    # We're going to *pretend* that the action we take here is the "correct" one,
    # and use it as the label. We then want to compute the gradient of our pretend
    # loss function comparing the *probability* that we had previously assigned to that action
    # against the action itself. When we do that, the gradient we end up computing measures
    # in which direction in parameter space we would have to tweak our model's parameters in
    # order to make *the thing we actually did* be more likely in the future. We need to compute
    # that gradient *just in case* the action ends up leading to a positive outcome, but since we
    # don't have the reward feedback yet, we shouldn't backprop yet. That is, we compute the gradient
    # just in case, but we won't use it until later, or maybe not at all. Whether we use the gradients
    # we've computed of our policy here depend on the rewards we receive in the future, which are
    # the closest approximation we have here to "labels".

    # Okay, let's implement this shit.

    # Probability of left (i.e., 0) is 1 if our action is 0, by definition. Similarly for right.
    y_true = 1.0 - tf.to_float(action)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y_true, logits = logits)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    grad_var_pairs = optimizer.compute_gradients(cross_entropy)

    make_placeholder = lambda gradient: tf.placeholder(tf.float32, shape = gradient.get_shape())

    gradients = [gradient for gradient, variable in grad_var_pairs]
    variables = [variable for gradient, variable in grad_var_pairs]

    # Make a set of feedable (gradient, variable) pairs using placeholders.
    # We'll use these to apply the policy gradients we've computed once it's time for the backprop step.
    gradient_placeholders = [make_placeholder(gradient) for gradient, variable in grad_var_pairs]
    feedable_policy_gradients = [*zip(gradient_placeholders, variables)]

    backpropagation = optimizer.apply_gradients(feedable_policy_gradients)

    # Now let's train our agent! :D

    n_training_iterations = 10000  # Total number of backpropagations.
    n_max_steps_per_episode = 1000 # Don't play the game forever even if we get good enough to.
    n_episodes_per_backprop = 2    # Number of episodes to run before applying the policy gradients.
    save_every = 100
    render_every = 20
    discount_rate = 0.95

    global_step = tf.train.get_or_create_global_step()
    increment_step = tf.assign_add(global_step, 1)

    checkpoint_dir = 'policy-gradients-model'
    sess, saver = initialize_session(checkpoint_dir)


    while True:

        iteration = sess.run(global_step)

        all_rewards = []
        all_gradients = []

        for episode in range(n_episodes_per_backprop):

            current_rewards = []
            current_gradients = []

            # reset the environment to prepare for a new episode, and get its first observation.
            obs = env.reset()

            for step in range(n_max_steps_per_episode):

                single_observation_batch = obs.reshape(1, -1) # or (1, n_inputs)

                # forward pass through our policy network to select an action

                action_val, gradients_val = sess.run(
                    [action, gradients],
                    feed_dict = {
                        inputs: single_observation_batch
                    }
                )

                # action_val.shape == (1, 1)
                # action_val will be either [[0]] or [[1]]

                action_selected = action_val[0][0]

                # take action, interact with the environment, and see what happens
                obs, reward, done, info = env.step(action_selected)

                if (iteration % render_every == 0) and (episode == 0):
                    env.render()
                else:
                    env.close()

                current_rewards.append(reward)
                current_gradients.append(gradients_val)

                if done:
                    break

            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)


        # Once we're here, we've got the rewards from enough episodes to perform a single backprop.
        # Time to update our policy using the gradients we collected in the episodes above.

        # Before we normalize the rewards for backprop, let's summarize their
        # raw values so we can see if they're changing over time during training.
        all_rewards_clearer = np.array([sum(ones_list) for ones_list in all_rewards])
        episode_rewards_avg = all_rewards_clearer.mean()
        episode_rewards_std = all_rewards_clearer.std()
        recent_reward_buffer.appendleft(episode_rewards_avg)

        recent_rewards_mean = np.mean(recent_reward_buffer)

        print(f"{iteration:04d}: Completed batch of {n_episodes_per_backprop} episodes. "
              f"Mean episode reward is {episode_rewards_avg:.02f}. "
              f"Rolling mean: {recent_rewards_mean:.02f}."
        )

        # Now let's discount and normalize the rewards in preparation for the big backprop step.
        all_rewards = discount_and_normalize_all_rewards(all_rewards, discount_rate)

        feed_dict = {}

        for variable_index, gradient_placeholder in enumerate(gradient_placeholders):

            # multiply each gradient by the score its action received, and average.

            weighted_gradients = np.array([
                reward * all_gradients[episode_index][step_index][variable_index]
                for episode_index, rewards in enumerate(all_rewards)
                for step_index, reward in enumerate(rewards)
            ])

            mean_gradients = np.mean(weighted_gradients, axis = 0)

            feed_dict[gradient_placeholder] = mean_gradients

        #lr = 0.01 if (recent_rewards_mean < 200) else 0.001
        #lr = 1 / recent_rewards_mean # now here's a weird idea...
        lr = 0.01 if (recent_rewards_mean < 400) else 0.005

        sess.run([backpropagation, increment_step], feed_dict = {**feed_dict, learning_rate: lr})

        if not iteration % save_every:
            saver.save(sess, f"{checkpoint_dir}/policy-gradients.ckpt")

        if recent_rewards_mean >= 0.8*ENVIRONMENT_WIN_CRITERION:
            render_every = 10

        if recent_rewards_mean >= ENVIRONMENT_WIN_CRITERION:
            print(f"Environment objective completed. Nice work.")
            break

        if iteration > n_training_iterations:
            print(f"Exceeded max number of training iterations without reaching goal. Exiting.")
            break


@numba.jit
def discount_rewards(rewards, discount_rate):

    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0

    for step, reward in reversed(list(enumerate(rewards))):
        cumulative_rewards = reward + (discount_rate * cumulative_rewards)
        discounted_rewards[step] = cumulative_rewards

    return discounted_rewards


@numba.jit
def discount_and_normalize_all_rewards(all_rewards, discount_rate):

    """
        Apply the reward baseline to make our reinforcement learning problem
        depend less on the rewards' implicit coordinate system and more on
        their magnitudes relative to each other. That way, we can decrease the
        probability of an action *even if it has a positive reward*, provided
        it's a less good reward than the average reward received in a given
        batch of our agent's experience.
    """

    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flattened_rewards = np.concatenate(all_discounted_rewards)
    global_avg = flattened_rewards.mean()
    global_std = flattened_rewards.std()
    normalize_reward_batch = lambda reward_batch: (reward_batch - global_avg) / global_std
    return [normalize_reward_batch(reward_batch) for reward_batch in all_discounted_rewards]


ENVIRONMENT_WIN_CRITERION = 10_000

gym.envs.register(
    id = 'CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10_000},
    reward_threshold = 10_000,
)

env = gym.make('CartPole-v2')

policy_gradients(env)


import os
import time
from collections import deque

import numpy as np
import tqdm
import random

from .env.ttt.DeepTicTacToe import DeepTicTacToe
from ..do_not_touch.deep_single_agent_with_discrete_actions_env_wrapper import Env5

import tensorflow as tf


class DeepQNetwork:
    """
    Contains the weights, structure of the q_network
    """
    pass


def test_nn(env, state_description_length, max_actions_count, q):
    win = 0
    lose = 0
    draw = 0

    for _ in tqdm.tqdm(range(1000)):
        env.reset()
        while not env.is_game_over():
            s = env.state_description()
            available_actions = env.available_actions_ids()

            chosen_action = predict(q, s.reshape((1, 3, 3, 1)), available_actions)
            env.act_with_action_id(chosen_action)

            if env.is_game_over():
                if env.is_win():
                    win += 1
                elif env.is_draw():
                    draw += 1
                elif env.is_loss():
                    lose += 1

    print("Win", win)
    print("Loss", lose)
    print("Draw", draw)


def predict(nn, state, available_actions):
    probabilities = np.squeeze(nn.predict(state))

    for i, p in enumerate(probabilities):
        if i not in available_actions:
            probabilities[i] = -10000

    return np.argmax(probabilities)


def episodic_semi_gradient_sarsa_on_tic_tac_toe_solo() -> DeepQNetwork:
    #tf.config.set_visible_devices([], 'GPU')
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
    Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """

    env = DeepTicTacToe()
    pre_warm = 100
    epsilon = 0.5
    gamma = 0.9999
    max_episodes_count = 200000
    print_every_n_episodes = 500

    win = 0
    lose = 0
    draw = 0

    state_description_length = env.state_description_length()
    max_actions_count = env.max_actions_count()
    q = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(128, kernel_size=(2, 2), input_shape=(3, 3, 1), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
            tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(9, activation="softmax"),
        ]
    )

    q.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mse)
    q.summary()

    if os.path.isfile("semi_gradient_sarsa.h5"):
        q.load_weights("semi_gradient_sarsa.h5")
        print("Restore model")
        #test_nn(env, state_description_length, max_actions_count, q)

    for episode_id in tqdm.tqdm(range(max_episodes_count)):
        if np.random.uniform(0.0, 1.0) > 0.5:
            env.reset_random()
        else:
            env.reset()

        # if episode_id == 1000:
        #     epsilon = 0.05
        if epsilon > 0.02:
            epsilon = epsilon * 0.9997

        while not env.is_game_over():
            s = env.state_description().reshape((1, 3, 3, 1))
            available_actions = env.available_actions_ids()

            if episode_id < pre_warm or np.random.uniform(0.0, 1.0) < epsilon:
                chosen_action = np.random.choice(available_actions)
            else:
                chosen_action = predict(q, s, available_actions)

            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score
            s_p = env.state_description().reshape((1, 3, 3, 1))

            if env.is_game_over():
                target = np.zeros(max_actions_count)
                target[chosen_action] = r
                q.train_on_batch(s, target.reshape(-1, 9))
                if env.is_win():
                    win += 1
                elif env.is_draw():
                    draw += 1
                elif env.is_loss():
                    lose += 1
                if episode_id % print_every_n_episodes == 0:
                    # env.view()
                    # print(f'Chosen action : {chosen_action} | score: {target}')
                    # print(f'Chosen action value : {chosen_action_q_value} [{all_q_values}]')
                    # print("Score:", "win" if env.is_win() else "lose" if env.is_loss() else "draw")
                    print("\nWin:", win, " | Lose :", lose, " | Draw:", draw, " /// Eps:", epsilon)
                    win = 0
                    draw = 0
                    lose = 0
                break

            next_available_actions = env.available_actions_ids()

            if episode_id < pre_warm or np.random.uniform(0.0, 1.0) < epsilon:
                next_chosen_action = np.random.choice(next_available_actions)
            else:
                next_chosen_action = None
                next_chosen_action_q_value = None
                q_values_ = np.squeeze(q.predict(s_p))
                for a in next_available_actions:
                    p = q_values_[a]
                    if next_chosen_action is None or next_chosen_action_q_value < p:
                        next_chosen_action = a
                        next_chosen_action_q_value = p

            next_chosen_action_q_value = np.squeeze(q.predict(s_p))[next_chosen_action]
            target = np.zeros(max_actions_count)
            target[next_chosen_action] = r + gamma * next_chosen_action_q_value
            q.train_on_batch(s, target.reshape(-1, 9))

        if episode_id % 1000 == 0:
            q.save_weights("semi_gradient_sarsa.h5")

    q.save_weights("semi_gradient_sarsa.h5")
    pass


def replay(memory, batch_size, gamma, model):
    min_batch = random.sample(memory, batch_size)
    states = np.empty((batch_size, 9))
    targets = np.empty((batch_size, 9))

    targets_f = model.predict(np.array([s[0] for s in min_batch]).reshape((batch_size, 3, 3, 1)))
    for i, (state, action, reward, next_state, done) in enumerate(min_batch):
        target = reward
        if not done:
            target = (reward + gamma *
                      np.amax(model.predict(next_state)[0]))

        target_f = targets_f[i]
        target_f[action] = target
        targets[i] = target_f
        states[i] = state

    model.fit(
        states,
        targets,
        batch_size=batch_size,
        epochs=1,
        verbose=0
    )


def deep_q_learning_on_tic_tac_toe_solo() -> DeepQNetwork:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
    Returns the optimal action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    tf.config.set_visible_devices([], 'GPU')
    env = DeepTicTacToe()
    pre_warm = 50
    epsilon = 0.6
    gamma = 0.995
    max_episodes_count = 50000
    print_every_n_episodes = 500
    batch_size = 64

    win = 0
    lose = 0
    draw = 0

    memory = deque(maxlen=300)

    state_description_length = env.state_description_length()
    max_actions_count = env.max_actions_count()
    q = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(128, kernel_size=(2, 2), input_shape=(3, 3, 1), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
            tf.keras.layers.Conv2D(64, kernel_size=(2, 2), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(9, activation="softmax"),
        ]
    )
    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)
    q.summary()

    if os.path.isfile("dql_ttt.h5"):
        q.load_weights("dql_ttt.h5")
        print("Restore model DQL")
        # test_nn(env, state_description_length, max_actions_count, q)

    i = 0
    for episode_id in tqdm.tqdm(range(max_episodes_count)):
        if epsilon > 0.1:
            epsilon *= 0.9999

        if np.random.uniform(0.0, 1.0) > 0.5:
            env.reset_random()
        else:
            env.reset()
        state = env.state_description().reshape((1, 3, 3, 1))

        while not env.is_game_over():
            i += 1
            available_actions = env.available_actions_ids()

            if episode_id < pre_warm or np.random.uniform(0.0, 1.0) < epsilon:
                action = np.random.choice(available_actions)
            else:
                action = predict(q, state, available_actions)

            env.act_with_action_id(action)
            reward = env.score()
            done = env.is_game_over()
            next_state = env.state_description().reshape((1, 3, 3, 1))
            memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                if env.is_win():
                    win += 1
                elif env.is_draw():
                    draw += 1
                elif env.is_loss():
                    lose += 1
                if episode_id % print_every_n_episodes == 0:
                    print("\nWin:", win, " | Lose :", lose, " | Draw:", draw, " /// Eps:", epsilon)
                    win = 0
                    draw = 0
                    lose = 0
                break

            if len(memory) > batch_size and i % 2 == 0:
                replay(memory, batch_size, gamma, q)

            if episode_id % 100 == 0:
                q.save_weights("dql_ttt.h5")

    pass


def episodic_semi_gradient_sarsa_on_pac_man() -> DeepQNetwork:
    """
    Creates a PacMan environment
    Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
    Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def deep_q_learning_on_pac_man() -> DeepQNetwork:
    """
    Creates a PacMan environment
    Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
    Returns the optimal action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def episodic_semi_gradient_sarsa_on_secret_env5() -> DeepQNetwork:
    """
    Creates a Secret Env 5 environment
    Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
    Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    tf.config.set_visible_devices([], 'GPU')
    env = Env5()
    pre_warm = 500
    epsilon = 0.5
    gamma = 0.9999
    max_episodes_count = 200000
    print_every_n_episodes = 500

    total_score = 0

    state_description_length = env.state_description_length()
    max_actions_count = env.max_actions_count()

    print(state_description_length, max_actions_count)
    q = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(128, input_shape=(state_description_length, ), activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    q.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mse)
    q.summary()

    if os.path.isfile("semi_gradient_sarsa_se.h5"):
        q.load_weights("semi_gradient_sarsa_se.h5")
        print("Restore model")
        # test_nn(env, state_description_length, max_actions_count, q)

    for episode_id in tqdm.tqdm(range(max_episodes_count)):
        env.reset()

        # if episode_id == 1000:
        #     epsilon = 0.05
        if epsilon > 0.02:
            epsilon = epsilon * 0.9997

        i = 0
        while not env.is_game_over() and i < (100 if (episode_id < pre_warm) else 200):
            i += 1
            s = env.state_description().reshape(-1, state_description_length)
            available_actions = env.available_actions_ids()

            if episode_id < pre_warm or np.random.uniform(0.0, 1.0) < epsilon:
                chosen_action = np.random.choice(available_actions)
            else:
                chosen_action = predict(q, s, available_actions)

            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score
            s_p = env.state_description().reshape(-1, state_description_length)

            if env.is_game_over():
                target = np.zeros(max_actions_count)
                target[chosen_action] = r
                q.train_on_batch(s, target.reshape(-1, max_actions_count))

                if episode_id % print_every_n_episodes == 0:
                    print("\nAvg. score:", int(total_score / 1000), "/// Eps:", epsilon)
                    total_score = 0
                break

            next_available_actions = env.available_actions_ids()

            if episode_id < pre_warm or np.random.uniform(0.0, 1.0) < epsilon:
                next_chosen_action = np.random.choice(next_available_actions)
            else:
                next_chosen_action = None
                next_chosen_action_q_value = None
                q_values_ = np.squeeze(q.predict(s_p))
                for a in next_available_actions:
                    p = q_values_[a]
                    if next_chosen_action is None or next_chosen_action_q_value < p:
                        next_chosen_action = a
                        next_chosen_action_q_value = p

            next_chosen_action_q_value = np.squeeze(q.predict(s_p))[next_chosen_action]
            target = np.zeros(max_actions_count)
            target[next_chosen_action] = r + gamma * next_chosen_action_q_value
            q.train_on_batch(s, target.reshape(-1, max_actions_count))

        if episode_id % 1000 == 0:
            q.save_weights("semi_gradient_sarsa_se.h5")

    q.save_weights("semi_gradient_sarsa_se.h5")
    pass


def deep_q_learning_on_secret_env5() -> DeepQNetwork:
    """
    Creates a Secret Env 5 environment
    Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
    Returns the optimal action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    _env = Env5()
    pass


def demo():
    #print(episodic_semi_gradient_sarsa_on_tic_tac_toe_solo())

    print(deep_q_learning_on_tic_tac_toe_solo())
    #
    # print(episodic_semi_gradient_sarsa_on_pac_man())
    # print(deep_q_learning_on_pac_man())
    #
    #print(episodic_semi_gradient_sarsa_on_secret_env5())
    # print(deep_q_learning_on_secret_env5())

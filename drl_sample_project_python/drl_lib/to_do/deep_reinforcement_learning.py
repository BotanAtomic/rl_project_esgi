import os

import numpy as np
import tqdm

from .env.ttt.DeepTicTacToe import DeepTicTacToe
from ..do_not_touch.deep_single_agent_with_discrete_actions_env_wrapper import Env5

import tensorflow as tf

class DeepQNetwork:
    """
    Contains the weights, structure of the q_network
    """
    pass


def episodic_semi_gradient_sarsa_on_tic_tac_toe_solo() -> DeepQNetwork:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
    Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = DeepTicTacToe()
    epsilon = 0.199
    gamma = 0.9
    max_episodes_count = 10000
    print_every_n_episodes = 10

    state_description_length = env.state_description_length()
    max_actions_count = env.max_actions_count()
    q = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh,
                              input_dim=(state_description_length + max_actions_count)),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])
    q.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)

    if os.path.isfile("semi_gradient_sarsa.h5"):
        q.load_weights("semi_gradient_sarsa.h5")
        print("Restore model")

    for episode_id in tqdm.tqdm(range(max_episodes_count)):
        env.reset()

        while not env.is_game_over():
            s = env.state_description()
            available_actions = env.available_actions_ids()

            chosen_action_q_value = None

            if np.random.uniform(0.0, 1.0) < epsilon:
                chosen_action = np.random.choice(available_actions)
            else:
                all_q_inputs = np.zeros((len(available_actions), state_description_length + max_actions_count))
                for i, a in enumerate(available_actions):
                    all_q_inputs[i] = np.hstack([s, tf.keras.utils.to_categorical(a, max_actions_count)])

                all_q_values = np.squeeze(q.predict(all_q_inputs))
                chosen_action = available_actions[np.argmax(all_q_values)]
                chosen_action_q_value = np.max(all_q_values)



            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score
            s_p = env.state_description()

            if env.is_game_over():
                target = r
                q_inputs = np.hstack([s, tf.keras.utils.to_categorical(chosen_action, max_actions_count)])
                q.train_on_batch(np.array([q_inputs]), np.array([target]))
                if episode_id % print_every_n_episodes == 0:
                    env.view()
                    print(f'Chosen action : {chosen_action}')
                    print(f'Chosen action value : {chosen_action_q_value}')
                    print("Score:", "win" if env.is_win() else "lose" if env.is_loss() else "draw")
                break

            next_available_actions = env.available_actions_ids()

            if np.random.uniform(0.0, 1.0) < epsilon:
                next_chosen_action = np.random.choice(next_available_actions)
            else:
                next_chosen_action = None
                next_chosen_action_q_value = None
                for a in next_available_actions:
                    q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(a, max_actions_count)])
                    q_value = q.predict(np.array([q_inputs]))[0][0]
                    if next_chosen_action is None or next_chosen_action_q_value < q_value:
                        next_chosen_action = a
                        next_chosen_action_q_value = q_value

            next_q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(next_chosen_action, max_actions_count)])
            next_chosen_action_q_value = q.predict(np.array([next_q_inputs]))[0][0]

            target = r + gamma * next_chosen_action_q_value

            q_inputs = np.hstack([s, tf.keras.utils.to_categorical(chosen_action, max_actions_count)])
            q.train_on_batch(np.array([q_inputs]), np.array([target]))

    q.save_weights("semi_gradient_sarsa.h5")
    pass


def deep_q_learning_on_tic_tac_toe_solo() -> DeepQNetwork:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
    Returns the optimal action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
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
    # TODO
    _env = Env5()
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
    print(episodic_semi_gradient_sarsa_on_tic_tac_toe_solo())
    # print(deep_q_learning_on_tic_tac_toe_solo())
    #
    # print(episodic_semi_gradient_sarsa_on_pac_man())
    # print(deep_q_learning_on_pac_man())
    #
    # print(episodic_semi_gradient_sarsa_on_secret_env5())
    # print(deep_q_learning_on_secret_env5())

from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .env.ttt.TicTacToe import TicTacToe
from ..do_not_touch.contracts import SingleAgentEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env3


def init_Q(state, env: SingleAgentEnv, Q: Dict[int, Dict[int, float]]):
    if state not in Q:
        Q[state] = {}
        for action in env.available_actions_ids():
            Q[state][action] = 0.0


def choose_action(state, epsilon, env: SingleAgentEnv, Q: Dict[int, Dict[int, float]]):
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(env.available_actions_ids(), 1)[0]
    else:
        action = max(Q[state], key=Q[state].get)

    return action


def sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()
    max_iteration = 50000
    test_at_step = 49000
    alpha = 0.2
    epsilon = 0.01
    gamma = 0.9999
    Q = {}

    win = 0
    loss = 0
    draw = 0

    average_rewards = []
    rewards_history = np.zeros(1000)

    decrease_epsilon = False
    decrease_rate = 100

    for _ in tqdm(range(max_iteration)):
        rewards_history[_ % 1000] = env.score()

        if _ % 1000 == 0 and _ > 0:
            average_rewards.append(np.mean(rewards_history))

        if decrease_epsilon and _ % (max_iteration / decrease_rate) == 0 and _ > 0:
            epsilon *= 0.99

        if _ >= test_at_step:
            epsilon = 0.0

        if _ >= test_at_step:
            env.reset()
        else:
            env.reset_random()

        state = env.state_id()
        init_Q(state, env, Q)
        action = choose_action(state, epsilon, env, Q)

        while not env.is_game_over():
            env.act_with_action_id(action)

            done = env.is_game_over()
            reward = env.score()

            if done:
                if env.is_win() and _ >= test_at_step:
                    win += 1
                elif env.is_loss() and _ >= test_at_step:
                    loss += 1
                elif env.is_draw() and _ >= test_at_step:
                    draw += 1

            state_prime = env.state_id()
            init_Q(state_prime, env, Q)
            action_prime = 0 if done else choose_action(state_prime, epsilon, env, Q)

            if not done:
                Q[state][action] += (alpha * (reward + gamma * Q[state_prime][action_prime] - Q[state][action]))
            else:
                Q[state][action] += (alpha * (reward - Q[state][action]))

            state = state_prime
            action = action_prime

    print("Win ", win, "/", win + loss + draw)
    print("Loss ", loss, "/", win + loss + draw)
    print("Draw ", draw, "/", win + loss + draw)
    plt.plot(average_rewards)
    plt.show()
    return PolicyAndActionValueFunction(pi={}, q=Q)


def q_learning_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()
    max_iteration = 50000
    test_at_step = 49000
    alpha = 0.2
    epsilon = 0.01
    gamma = 0.9999
    Q = {}

    win = 0
    loss = 0
    draw = 0

    average_rewards = []
    rewards_history = np.zeros(1000)

    decrease_epsilon = False
    decrease_rate = 50

    if decrease_epsilon:
        epsilon = 0.1  # we start with a big epsilon

    for _ in tqdm(range(max_iteration)):
        rewards_history[_ % 1000] = env.score()

        if _ % 1000 == 0 and _ > 0:
            average_rewards.append(np.mean(rewards_history))

        if decrease_epsilon and _ % (max_iteration / decrease_rate) == 0 and _ > 0:
            epsilon *= 0.99

        if _ >= test_at_step:
            epsilon = 0.0

        if _ >= test_at_step:
            env.reset()
        else:
            env.reset_random()

        state = env.state_id()
        init_Q(state, env, Q)

        while not env.is_game_over():
            action = choose_action(state, epsilon, env, Q)

            env.act_with_action_id(action)

            done = env.is_game_over()
            reward = env.score()
            state_prime = env.state_id()
            init_Q(state_prime, env, Q)

            if done:
                if env.is_win() and _ >= test_at_step:
                    win += 1
                elif env.is_loss() and _ >= test_at_step:
                    loss += 1
                elif env.is_draw() and _ >= test_at_step:
                    draw += 1

            if not done:
                Q[state][action] += alpha * (reward + (gamma * max(list(Q[state_prime].values())) - Q[state][action]))
            else:
                Q[state][action] += alpha * (reward - Q[state][action])

            state = state_prime

    print("Win ", win, "/", win + loss + draw)
    print("Loss ", loss, "/", win + loss + draw)
    print("Draw ", draw, "/", win + loss + draw)

    plt.plot(average_rewards)
    plt.show()
    return PolicyAndActionValueFunction(pi={}, q=Q)


def expected_sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()
    max_iteration = 50000
    test_at_step = 49000
    alpha = 0.2
    epsilon = 0.01
    gamma = 0.9999
    Q = {}

    win = 0
    loss = 0
    draw = 0

    average_rewards = []
    rewards_history = np.zeros(1000)

    decrease_epsilon = False
    decrease_rate = 50

    if decrease_epsilon:
        epsilon = 0.1  # we start with a big epsilon

    for _ in tqdm(range(max_iteration)):
        rewards_history[_ % 1000] = env.score()

        if _ % 1000 == 0 and _ > 0:
            average_rewards.append(np.mean(rewards_history))

        if decrease_epsilon and _ % (max_iteration / decrease_rate) == 0 and _ > 0:
            epsilon *= 0.99

        if _ >= test_at_step:
            epsilon = 0.0

        if _ >= test_at_step:
            env.reset()
        else:
            env.reset_random()

        state = env.state_id()
        init_Q(state, env, Q)

        while not env.is_game_over():
            action = choose_action(state, epsilon, env, Q)

            env.act_with_action_id(action)

            done = env.is_game_over()
            reward = env.score()
            state_prime = env.state_id()
            init_Q(state_prime, env, Q)

            if done:
                if env.is_win() and _ >= test_at_step:
                    win += 1
                elif env.is_loss() and _ >= test_at_step:
                    loss += 1
                elif env.is_draw() and _ >= test_at_step:
                    draw += 1

            if not done:
                Q[state][action] += alpha * (reward + (gamma * max(list(Q[state_prime].values())) - Q[state][action]))
            else:
                Q[state][action] += alpha * (reward - Q[state][action])

            state = state_prime

    print("Win ", win, "/", win + loss + draw)
    print("Loss ", loss, "/", win + loss + draw)
    print("Draw ", draw, "/", win + loss + draw)

    plt.plot(average_rewards)
    plt.show()
    return PolicyAndActionValueFunction(pi={}, q=Q)


def sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    max_iteration = 10000
    test_at_step = int(max_iteration * 0.9)
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.999
    Q = {}

    average_rewards = []
    rewards_history = np.zeros(1000)

    decrease_epsilon = False
    decrease_rate = 100

    for _ in tqdm(range(max_iteration)):
        rewards_history[_ % 1000] = env.score()

        if _ % 1000 == 0 and _ > 0:
            average_rewards.append(np.mean(rewards_history))

        if decrease_epsilon and _ % (max_iteration / decrease_rate) == 0 and _ > 0:
            epsilon *= 0.99

        if _ >= test_at_step:
            epsilon = 0.0

        if _ >= test_at_step:
            env.reset()
        else:
            env.reset_random()

        state = env.state_id()
        init_Q(state, env, Q)
        action = choose_action(state, epsilon, env, Q)

        while not env.is_game_over():
            env.act_with_action_id(action)

            done = env.is_game_over()
            reward = env.score()

            state_prime = env.state_id()
            init_Q(state_prime, env, Q)
            action_prime = 0 if done else choose_action(state_prime, epsilon, env, Q)

            if not done:
                Q[state][action] += (alpha * (reward + gamma * Q[state_prime][action_prime] - Q[state][action]))
            else:
                Q[state][action] += (alpha * (reward - Q[state][action]))

            state = state_prime
            action = action_prime

    plt.plot(average_rewards[:(test_at_step)//1000])
    plt.show()
    return PolicyAndActionValueFunction(pi={}, q=Q)
    pass


def q_learning_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def expected_sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def demo():
    # sarsa_on_tic_tac_toe_solo()
    # q_learning_on_tic_tac_toe_solo()
    # print(expected_sarsa_on_tic_tac_toe_solo())
    #
    print(sarsa_on_secret_env3())
    # print(q_learning_on_secret_env3())
    # print(expected_sarsa_on_secret_env3())

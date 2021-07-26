import numpy as np
import tqdm
from matplotlib import pyplot as plt

from .agent.Reinforce import PGAgent
from .env.ttt.DeepTicTacToe import DeepTicTacToe
from ..do_not_touch.deep_single_agent_with_discrete_actions_env_wrapper import Env5

import tensorflow as tf


class DeepPiNetwork:
    """
    Contains the weights, structure of the pi_network (policy)
    """
    # TODO


class DeepVNetwork:
    """
    Contains the weights, structure of the v_network (baseline)
    """

    pass


def reinforce_on_tic_tac_toe_solo() -> DeepPiNetwork:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a REINFORCE Algorithm in order to find the optimal policy
    Returns the optimal policy network (Pi(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = DeepTicTacToe()
    state_size = env.state_description_length()
    action_size = env.max_actions_count()
    agent = PGAgent(state_size, action_size)

    max_episodes = 40000

    win = 0
    lose = 0
    draw = 0

    wins = []
    loses = []
    draws = []

    for episode_id in tqdm.tqdm(range(max_episodes)):
        env.reset()
        while not env.is_game_over():
            state = env.state_description().reshape(-1, 9).copy()
            available_actions = env.available_actions_ids()
            action = agent.act(state, available_actions)
            env.act_with_action_id(action)
            reward = env.score()
            done = env.is_game_over()
            agent.memorize(state, action, reward)

            if done:
                agent.update_network()
                if env.is_win():
                    win += 1
                elif env.is_draw():
                    draw += 1
                elif env.is_loss():
                    lose += 1
                if episode_id % 100 == 0 and episode_id > 0:
                    print("\nWin:", win, " | Lose :", lose, " | Draw:", draw)
                    wins.append(win)
                    loses.append(lose)
                    draws.append(draw)
                    win = 0
                    draw = 0
                    lose = 0
    plt.plot(wins, label='Win')
    plt.plot(loses, label='Lose')
    plt.plot(draws, label='Draw')
    plt.legend(['Win', 'Lose', 'Draw'])
    plt.show()


def reinforce_with_baseline_on_tic_tac_toe_solo() -> (DeepPiNetwork, DeepVNetwork):
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a REINFORCE with Baseline algorithm  in order to find the optimal policy and its value function
    Returns the optimal policy network (Pi(w)(s,a)) and its value function (V(w)(s))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """


def reinforce_on_pac_man() -> DeepPiNetwork:
    """
    Creates a PacMan environment
    Launches a REINFORCE Algorithm in order to find the optimal policy
    Returns the optimal policy network (Pi(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def reinforce_with_baseline_on_pac_man() -> (DeepPiNetwork, DeepVNetwork):
    """
    Creates a PacMan environment
    Launches a REINFORCE with Baseline algorithm in order to find the optimal policy and its value function
    Returns the optimal policy network (Pi(w)(s,a)) and its value function (V(w)(s))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def reinforce_on_secret_env_5() -> DeepPiNetwork:
    """
    Creates a Secret Env 5 environment
    Launches a REINFORCE Algorithm in order to find the optimal policy
    Returns the optimal policy network (Pi(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    _env = Env5()
    pass


def reinforce_with_baseline_on_secret_env_5() -> (DeepPiNetwork, DeepVNetwork):
    """
    Creates a Secret Env 5 environment
    Launches a REINFORCE with Baseline algorithm  in order to find the optimal policy and its value function
    Returns the optimal policy network (Pi(w)(s,a)) and its value function (V(w)(s))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    _env = Env5()
    pass


def demo():
    print(reinforce_on_tic_tac_toe_solo())
    # print(reinforce_with_baseline_on_tic_tac_toe_solo())
    #
    # print(reinforce_on_pac_man())
    # print(reinforce_with_baseline_on_pac_man())
    #
    # print(reinforce_on_secret_env_5())
    # print(reinforce_with_baseline_on_secret_env_5())

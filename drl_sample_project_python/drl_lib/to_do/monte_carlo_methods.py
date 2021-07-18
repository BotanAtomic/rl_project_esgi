from matplotlib import pyplot as plt

from .env.ttt.TicTacToe import TicTacToe
from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env2
from tqdm import tqdm
import numpy as np


def monte_carlo_es_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = TicTacToe()
    max_iterations = 30000
    gamma = 0.99999

    pi = {}
    q = {}
    returns = {}

    average_rewards = []
    rewards_history = np.zeros(100)

    for _ in tqdm(range(max_iterations)):
        rewards_history[_ % 100] = env.score()

        if _ % 100 == 0:
            average_rewards.append(np.mean(rewards_history))

        env.reset()
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    returns[s][a] = []

            chosen_action = np.random.choice(
                list(pi[s].keys()),
                1,
                False,
                p=list(pi[s].values())
            )[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

        G = 0

        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]
            found = False
            for p_s, p_a in zip(S[:t], A[:t]):
                if s_t == p_s and a_t == p_a:
                    found = True
                    break
            if found:
                continue
            returns[s_t][a_t].append(G)
            q[s_t][a_t] = np.mean(returns[s_t][a_t])
            pi[s_t].clear()
            pi[s_t][max(q[s_t], key=q[s_t].get)] = 1.0

    plt.plot(average_rewards)
    plt.show()
    return PolicyAndActionValueFunction(pi, q)


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()
    max_iterations = 100000
    epsilon = 0.05
    gamma = 0.99999

    pi = {}
    q = {}
    returns = {}

    average_rewards = []
    rewards_history = np.zeros(100)

    for _ in tqdm(range(max_iterations)):
        rewards_history[_ % 100] = env.score()

        if _ % 100 == 0:
            average_rewards.append(np.mean(rewards_history))

        env.reset()
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    returns[s][a] = []

            chosen_action = np.random.choice(
                list(pi[s].keys()),
                1,
                False,
                p=list(pi[s].values())
            )[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

        G = 0

        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]
            found = False
            for p_s, p_a in zip(S[:t], A[:t]):
                if s_t == p_s and a_t == p_a:
                    found = True
                    break
            if found:
                continue
            returns[s_t][a_t].append(G)
            q[s_t][a_t] = np.mean(returns[s_t][a_t])
            optimal_a_t = max(q[s_t], key=q[s_t].get)
            available_actions_t_count = len(q[s_t])
            for a_key, q_s_a in q[s_t].items():
                if a_key == optimal_a_t:
                    pi[s_t][a_key] = 1 - epsilon + epsilon / available_actions_t_count
                else:
                    pi[s_t][a_key] = epsilon / available_actions_t_count

    plt.plot(average_rewards)
    plt.show()
    return PolicyAndActionValueFunction(pi, q)


def off_policy_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = TicTacToe()
    max_iterations = 30000
    gamma = 0.99999

    q = {}
    c = {}
    pi = {}
    b = {}

    for _ in tqdm(range(max_iterations)):
        env.reset()
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                b[s] = {}
                q[s] = {}
                c[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    b[s][a] = 1.0 / len(available_actions)
                    c[s][a] = 0.0

            chosen_action = np.random.choice(
                list(b[s].keys()),
                1,
                False,
                p=list(b[s].values())
            )[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

        G = 0.0
        W = 1.0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]

            c[s_t][a_t] += W
            q[s_t][a_t] += (W / c[s_t][a_t]) * (G - q[s_t][a_t])
            max_q = max(q[s_t], key=q[s_t].get)
            pi[s_t][max_q] = 1.0
            if a_t != max_q:
                break
            W = W * (1/b[s_t][a_t])

    return PolicyAndActionValueFunction(pi, q)


def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = Env2()
    max_iterations = 30000
    gamma = 0.99999

    pi = {}
    q = {}
    returns = {}

    average_rewards = []
    rewards_history = np.zeros(100)

    for _ in tqdm(range(max_iterations)):
        rewards_history[_ % 100] = env.score()

        if _ % 100 == 0:
            average_rewards.append(np.mean(rewards_history))

        env.reset()
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    returns[s][a] = []

            chosen_action = np.random.choice(
                list(pi[s].keys()),
                1,
                False,
                p=list(pi[s].values())
            )[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

        G = 0

        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]
            found = False
            for p_s, p_a in zip(S[:t], A[:t]):
                if s_t == p_s and a_t == p_a:
                    found = True
                    break
            if found:
                continue
            returns[s_t][a_t].append(G)
            q[s_t][a_t] = np.mean(returns[s_t][a_t])
            pi[s_t].clear()
            pi[s_t][max(q[s_t], key=q[s_t].get)] = 1.0

    plt.plot(average_rewards)
    plt.show()
    return PolicyAndActionValueFunction(pi, q)


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    max_iterations = 30000
    epsilon = 0.05
    gamma = 0.99999

    pi = {}
    q = {}
    returns = {}

    average_rewards = []
    rewards_history = np.zeros(100)

    for _ in tqdm(range(max_iterations)):
        rewards_history[_ % 100] = env.score()

        if _ % 100 == 0:
            average_rewards.append(np.mean(rewards_history))

        env.reset()
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    returns[s][a] = []

            chosen_action = np.random.choice(
                list(pi[s].keys()),
                1,
                False,
                p=list(pi[s].values())
            )[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

        G = 0

        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]
            found = False
            for p_s, p_a in zip(S[:t], A[:t]):
                if s_t == p_s and a_t == p_a:
                    found = True
                    break
            if found:
                continue
            returns[s_t][a_t].append(G)
            q[s_t][a_t] = np.mean(returns[s_t][a_t])
            optimal_a_t = max(q[s_t], key=q[s_t].get)
            available_actions_t_count = len(q[s_t])
            for a_key, q_s_a in q[s_t].items():
                if a_key == optimal_a_t:
                    pi[s_t][a_key] = 1 - epsilon + epsilon / available_actions_t_count
                else:
                    pi[s_t][a_key] = epsilon / available_actions_t_count

    plt.plot(average_rewards)
    plt.show()
    return PolicyAndActionValueFunction(pi, q)


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    max_iterations = 30000
    gamma = 0.99999

    q = {}
    c = {}
    pi = {}
    b = {}

    for _ in tqdm(range(max_iterations)):
        env.reset()
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                b[s] = {}
                q[s] = {}
                c[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    b[s][a] = 1.0 / len(available_actions)
                    c[s][a] = 0.0

            chosen_action = np.random.choice(
                list(b[s].keys()),
                1,
                False,
                p=list(b[s].values())
            )[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

        G = 0.0
        W = 1.0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]

            c[s_t][a_t] += W
            q[s_t][a_t] += (W / c[s_t][a_t]) * (G - q[s_t][a_t])
            max_q = max(q[s_t], key=q[s_t].get)
            pi[s_t][max_q] = 1.0
            if a_t != max_q:
                break
            W = W * (1 / b[s_t][a_t])

    return PolicyAndActionValueFunction(pi, q)


def demo():
    print(monte_carlo_es_on_tic_tac_toe_solo())
    #print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())
    #print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

    #print(monte_carlo_es_on_secret_env2())
    #print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    #print(off_policy_monte_carlo_control_on_secret_env2())

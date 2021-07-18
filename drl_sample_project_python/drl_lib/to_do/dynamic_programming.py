import random

import numpy.random

from .env.gridworld.GridWorldMDP import GridWorldMDP
from .env.gridworld.GridWorldSingleAgent import GridWorld
from .env.lineworld.LineWorldMDP import LineWorldMDP
from ..do_not_touch.contracts import MDPEnv
from ..do_not_touch.mdp_env_wrapper import Env1
from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
from typing import Dict

import numpy as np


def random_policy_line_world(env: MDPEnv) -> Dict[int, Dict[int, float]]:
    pi: Dict[int, Dict[int, float]] = {}

    for s in env.states():
        if not env.is_state_terminal(s):
            pi[s] = {}
            random_values = np.random.dirichlet(np.ones(len(env.actions())), size=1)
            for random_value in random_values:
                for i, r in zip(env.actions(), random_value):
                    pi[s][i] = r
    return pi


def evaluate_policy(env: MDPEnv, pi: Dict[int, Dict[int, float]], gamma: float, theta: float,
                    max_iterations: int) -> ValueFunction:
    S = env.states()
    V: Dict[int, float] = {}

    for s in S:
        V[s] = 0.0

    for _ in range(max_iterations):
        delta = 0
        for s in S:
            v = V[s]
            V[s] = 0

            if env.is_state_terminal(s):
                continue

            for action in env.actions():
                for next_state in S:
                    for r_index, r in enumerate(env.rewards()):
                        V[s] += pi[s][action] * env.transition_probability(s, action, next_state, r_index) \
                                * (r + gamma * V[next_state])

            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break
    return V


def policy_evaluation_on_line_world() -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    max_iterations = 10_000
    theta = 0.001
    gamma = 0.9
    env = LineWorldMDP(7)

    pi = random_policy_line_world(env)

    return evaluate_policy(env, pi, gamma, theta, max_iterations)


def policy_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    max_iterations = 10_000
    theta = 0.001
    gamma = 0.99
    env = LineWorldMDP(7)

    V: Dict[int, float] = {}
    S = env.states()

    pi: Dict[int, Dict[int, float]] = random_policy_line_world(env)

    policy_stable = False

    while not policy_stable:
        policy_stable = True

        V = evaluate_policy(env, pi, gamma, theta, max_iterations)

        for s in env.states():
            if env.is_state_terminal(s):
                continue

            old_action = pi[s].copy()

            for action in env.actions():
                pi[s][action] = 0

            best_pi = np.zeros(len(env.actions()))
            for action in env.actions():
                for next_state in S:
                    for r_index, r in enumerate(env.rewards()):
                        best_pi[action] += env.transition_probability(s, action, next_state, r_index) \
                                           * (r + gamma * V[next_state])

            best = np.argmax(best_pi)
            pi[s][int(best)] = best_pi[best]

            if old_action != pi[s]:
                policy_stable = False

        if policy_stable:
            break

    for s in S:
        if not env.is_state_terminal(s):
            assert (pi[s][1] > 0.85)

    return PolicyAndValueFunction(pi, V)


def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    max_iterations = 10_000
    theta = 0.001
    gamma = 0.99
    env = LineWorldMDP(7)

    V: Dict[int, float] = {}
    S = env.states()

    for s in S:
        V[s] = 0.0

    for _ in range(max_iterations):
        delta = 0

        for s in S:
            v = V[s]
            best_pi = np.zeros(len(env.actions()))

            for action in env.actions():
                for next_state in S:
                    for r_index, r in enumerate(env.rewards()):
                        best_pi[action] += env.transition_probability(s, action, next_state, r_index) \
                                           * (r + gamma * V[next_state])

            V[s] = max(best_pi)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    pi: Dict[int, Dict[int, float]] = {}

    for s in S:
        if env.is_state_terminal(s):
            continue

        pi[s] = {}

        best_pi = np.zeros(len(env.actions()))
        for action in env.actions():
            for next_state in S:
                for r_index, r in enumerate(env.rewards()):
                    best_pi[action] += env.transition_probability(s, action, next_state, r_index) \
                                       * (r + gamma * V[next_state])
        best = np.argmax(best_pi)
        pi[s][int(best)] = best_pi[best]

    for s in S:
        if not env.is_state_terminal(s):
            assert (pi[s][1] > 0.85)

    return PolicyAndValueFunction(pi, V)


def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    max_iterations = 10_000

    gridWorld = GridWorld(5, 5, 1000)
    env = GridWorldMDP(gridWorld)
    theta = 0.001
    gamma = 0.9
    pi = random_policy_line_world(env)
    return evaluate_policy(env, pi, gamma, theta, max_iterations)


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    max_iterations = 10_000

    gridWorld = GridWorld(5, 5, 1000)
    env = GridWorldMDP(gridWorld)
    theta = 0.001
    gamma = 0.9

    V: Dict[int, float] = {}
    S = env.states()

    pi: Dict[int, Dict[int, float]] = random_policy_line_world(env)

    policy_stable = False

    while not policy_stable:
        policy_stable = True

        V = evaluate_policy(env, pi, gamma, theta, max_iterations)

        for s in env.states():
            if env.is_state_terminal(s):
                continue

            old_action = pi[s].copy()

            for action in env.actions():
                pi[s][action] = 0

            best_pi = np.zeros(len(env.actions()))
            for action in env.actions():
                for next_state in S:
                    for r_index, r in enumerate(env.rewards()):
                        best_pi[action] += env.transition_probability(s, action, next_state, r_index) \
                                           * (r + gamma * V[next_state])

            best = np.argmax(best_pi)
            pi[s][int(best)] = best_pi[best]

            if old_action != pi[s]:
                policy_stable = False

        if policy_stable:
            break

    return PolicyAndValueFunction(pi, V)


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """

    gridWorld = GridWorld(5, 5, 1000)
    env = GridWorldMDP(gridWorld)
    max_iterations = 10_000
    theta = 0.001
    gamma = 0.99

    V: Dict[int, float] = {}
    S = env.states()

    for s in S:
        V[s] = 0.0

    for _ in range(max_iterations):
        delta = 0

        for s in S:
            v = V[s]
            best_pi = np.zeros(len(env.actions()))

            for action in env.actions():
                for next_state in S:
                    for r_index, r in enumerate(env.rewards()):
                        best_pi[action] += env.transition_probability(s, action, next_state, r_index) \
                                           * (r + gamma * V[next_state])

            V[s] = max(best_pi)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    pi: Dict[int, Dict[int, float]] = {}

    for s in S:
        if env.is_state_terminal(s):
            continue

        pi[s] = {}

        best_pi = np.zeros(len(env.actions()))
        for action in env.actions():
            for next_state in S:
                for r_index, r in enumerate(env.rewards()):
                    best_pi[action] += env.transition_probability(s, action, next_state, r_index) \
                                       * (r + gamma * V[next_state])
        best = np.argmax(best_pi)
        pi[s][int(best)] = best_pi[best]

    return PolicyAndValueFunction(pi, V)


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    max_iterations = 10_000
    theta = 0.001
    gamma = 0.9
    pi = random_policy_line_world(env)
    return evaluate_policy(env, pi, gamma, theta, max_iterations)


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """

    max_iterations = 10_000
    theta = 0.001
    gamma = 0.99
    env = Env1()

    V: Dict[int, float] = {}
    S = env.states()

    pi: Dict[int, Dict[int, float]] = random_policy_line_world(env)

    policy_stable = False

    while not policy_stable:
        policy_stable = True

        V = evaluate_policy(env, pi, gamma, theta, max_iterations)

        for s in env.states():
            if env.is_state_terminal(s):
                continue

            old_action = pi[s].copy()

            for action in env.actions():
                pi[s][action] = 0

            best_pi = np.zeros(len(env.actions()))
            for action in env.actions():
                for next_state in S:
                    for r_index, r in enumerate(env.rewards()):
                        best_pi[action] += env.transition_probability(s, action, next_state, r_index) \
                                           * (r + gamma * V[next_state])

            best = np.argmax(best_pi)
            pi[s][int(best)] = best_pi[best]

            if old_action != pi[s]:
                policy_stable = False

        if policy_stable:
            break

    return PolicyAndValueFunction(pi, V)


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    max_iterations = 10_000
    theta = 0.001
    gamma = 0.99

    V: Dict[int, float] = {}
    S = env.states()

    for s in S:
        V[s] = 0.0

    for _ in range(max_iterations):
        delta = 0

        for s in S:
            v = V[s]
            best_pi = np.zeros(len(env.actions()))

            for action in env.actions():
                for next_state in S:
                    for r_index, r in enumerate(env.rewards()):
                        best_pi[action] += env.transition_probability(s, action, next_state, r_index) \
                                           * (r + gamma * V[next_state])

            V[s] = max(best_pi)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    pi: Dict[int, Dict[int, float]] = {}

    for s in S:
        if env.is_state_terminal(s):
            continue

        pi[s] = {}

        best_pi = np.zeros(len(env.actions()))
        for action in env.actions():
            for next_state in S:
                for r_index, r in enumerate(env.rewards()):
                    best_pi[action] += env.transition_probability(s, action, next_state, r_index) \
                                       * (r + gamma * V[next_state])
        best = np.argmax(best_pi)
        pi[s][int(best)] = best_pi[best]

    return PolicyAndValueFunction(pi, V)


def demo():
    print(policy_evaluation_on_line_world())
    print(policy_iteration_on_line_world())
    print(value_iteration_on_line_world())

    print(policy_evaluation_on_grid_world())
    print(policy_iteration_on_grid_world())
    print(value_iteration_on_grid_world())

    print(policy_evaluation_on_secret_env1())
    print(policy_iteration_on_secret_env1())
    print(value_iteration_on_secret_env1())

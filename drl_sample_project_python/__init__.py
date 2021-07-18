from drl_lib.to_do.env.ttt.TicTacToe import TicTacToe
import numpy as np

env = TicTacToe()

average_score = 0.0

for i in range(10_000):
    while not env.is_game_over():
        actions = env.available_actions_ids()
        if len(actions) > 0:
            env.act_with_action_id(np.random.choice(actions))

    average_score += env.score()
    env.reset()
    print("OK", end='\r')


print(average_score)


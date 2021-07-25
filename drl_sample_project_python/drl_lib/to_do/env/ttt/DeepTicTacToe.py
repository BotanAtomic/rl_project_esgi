import numpy as np

from drl_lib.do_not_touch.contracts import DeepSingleAgentWithDiscreteActionsEnv
import tensorflow as tf

EMPTY = 0.0
X = 1.0
O = -1.0

# 0 1 2
# 3 4 5
# 6 7 8

WIN_POSITIONS = [[0, 1, 2], [2, 5, 8], [6, 7, 8], [6, 3, 0], [0, 4, 8], [6, 4, 2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8]]

SYMBOLS = [' ', 'X', 'O']

WIN = 1.0
DRAW = 0.2
LOSE = -1.0

class DeepTicTacToe(DeepSingleAgentWithDiscreteActionsEnv):

    def __init__(self):
        self.game_over = False
        self.turn = 0
        self.current_score = 0.0
        self.map = np.zeros(9)
        self.reset()

    def state_description(self) -> np.ndarray:
        return self.map

    def state_description_length(self) -> int:
        return 9

    def max_actions_count(self) -> int:
        return 9

    def is_game_over(self) -> bool:
        return self.game_over

    def is_win(self) -> bool:
        return self.score() == WIN

    def is_loss(self) -> bool:
        return self.score() == LOSE

    def is_draw(self) -> bool:
        return self.score() == DRAW

    def had_win(self, side: float) -> bool:
        for win_position in WIN_POSITIONS:
            win = True
            for pos in win_position:
                if self.map[pos] != side:
                    win = False
                    break
            if win:
                return True

    def act_with_action_id(self, action_id: int):
        assert (len(self.map) > action_id)
        assert (self.map[action_id] == EMPTY)
        assert not self.game_over

        self.map[action_id] = X

        if self.had_win(X):
            self.game_over = True
            self.current_score = WIN
            return

        actions = self.available_actions_ids()

        if len(actions) == 0:
            self.game_over = True
            self.current_score = DRAW
            return

        self.map[np.random.choice(actions)] = O

        if self.had_win(O):
            self.game_over = True
            self.current_score = LOSE
            return

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        actions = []
        for i in range(9):
            if self.map[i] == EMPTY:
                actions.append(i)

        return np.array(actions)

    def reset(self):
        self.game_over = False
        self.turn = 0
        self.current_score = 0.0
        for i in range(9):
            self.map[i] = 0

    def reset_random(self):
        self.reset()
        turns = np.random.randint(0, 8)
        for i in range(turns):
            if not self.is_game_over():
                self.act_with_action_id(np.random.choice(self.available_actions_ids()))

        if self.is_game_over():
            self.reset_random()

    def view(self):
        print("_______")
        for i in range(9):
            print('|', end=SYMBOLS[int(self.map[i])])
            if (i + 1) % 3 == 0 and i > 0:
                print('|')
        print("-------")

import numpy as np
from drl_lib.do_not_touch.contracts import SingleAgentEnv


class GridWorld(SingleAgentEnv):
    def __init__(self, rows: int, columns: int, max_steps: int):
        assert (rows >= 3)
        assert (columns >= 3)

        self.rows = rows
        self.columns = columns
        self.agent_pos = 0
        self.game_over = False
        self.current_score = 0.0
        self.max_steps = max_steps
        self.current_step = 0
        self.lose_pos = columns - 1
        self.win_pos = columns * rows - 1
        self.reset()

    def state_id(self) -> int:
        return self.agent_pos

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert (not self.game_over)
        assert (action_id == 0 or action_id == 1 or action_id == 2 or action_id == 3)

        if action_id == 0:
            self.agent_pos -= 1
        elif action_id == 1:
            self.agent_pos += 1
        elif action_id == 2:
            self.agent_pos -= self.columns
        else:
            self.agent_pos += self.columns

        if self.agent_pos == self.lose_pos:
            self.game_over = True
            self.current_score = -1

        elif self.agent_pos == self.win_pos:
            self.game_over = True
            self.current_score = 1

        self.current_step += 1

        if self.current_step >= self.max_steps:
            self.game_over = True

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        if self.game_over:
            return np.array([], dtype=np.int)
        res = []
        if self.agent_pos % self.columns != 0:
            res.append(0)

        if self.agent_pos % self.columns != self.rows - 1:
            res.append(1)

        if self.agent_pos >= self.columns:
            res.append(2)

        if self.agent_pos <= self.win_pos - self.columns:
            res.append(3)

        return np.array(res)

    def reset(self):
        nb_cases = self.rows * self.columns
        if (nb_cases // 2) == self.win_pos or (nb_cases // 2) == self.lose_pos:
            self.agent_pos = (nb_cases // 2) - 1
        else:
            self.agent_pos = nb_cases // 2
        self.game_over = False
        self.current_step = 0
        self.current_score = 0.0

    def reset_random(self):
        possible_pos = np.arange(self.rows * self.columns)
        np.delete(possible_pos, self.win_pos)
        np.delete(possible_pos, self.lose_pos)
        self.agent_pos = np.random.choice(possible_pos, 1)

        self.game_over = False
        self.current_step = 0
        self.current_score = 0.0

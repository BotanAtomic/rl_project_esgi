import numpy as np

from drl_lib.do_not_touch.contracts import SingleAgentEnv


class LineWorld(SingleAgentEnv):
    def __init__(self, cell_count: int, max_steps: int):
        assert (cell_count >= 3)
        self.cell_count = cell_count
        self.agent_pos = 0
        self.game_over = False
        self.current_score = 0.0
        self.max_steps = max_steps
        self.current_step = 0
        self.reset()

    def state_id(self) -> int:
        return self.agent_pos

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert (not self.game_over)
        assert (action_id == 0 or action_id == 1)
        if action_id == 0:
            self.agent_pos -= 1
        else:
            self.agent_pos += 1

        if self.agent_pos == 0:
            self.game_over = True
            self.current_score = -1
        elif self.agent_pos == self.cell_count - 1:
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
        return np.array([0, 1])  # 0: Left, 1: Right

    def reset(self):
        self.agent_pos = self.cell_count // 2
        self.game_over = False
        self.current_step = 0
        self.current_score = 0.0

    def reset_random(self):
        self.agent_pos = np.random.randint(1, self.cell_count - 1)
        self.game_over = False
        self.current_step = 0
        self.current_score = 0.0

import numpy as np
from drl_lib.do_not_touch.contracts import MDPEnv
from drl_lib.to_do.env.gridworld.GridWorldSingleAgent import GridWorld


class GridWorldMDP(MDPEnv):
    def __init__(self, grid_world: GridWorld):
        self.grid_world = grid_world

    def states(self) -> np.ndarray:
        return np.arange(self.grid_world.columns * self.grid_world.rows)

    def actions(self) -> np.ndarray:
        return np.arange(4)

    def rewards(self) -> np.ndarray:
        return np.array([-1, 0, 1])

    def is_state_terminal(self, s: int) -> bool:
        if s == self.grid_world.lose_pos or s == self.grid_world.win_pos:
            return True
        return False

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        assert (s in self.states())
        assert (a in self.actions())
        assert (s_p in self.states())
        r = self.rewards()[r]
        assert (r in self.rewards())

        if self.is_state_terminal(s):
            return 0.0

        if a == 0:
            if s % self.grid_world.columns != 0 and s_p == s - 1:
                if s_p == self.grid_world.lose_pos and r == -1.0:
                    return 1.0
                if s_p == self.grid_world.win_pos and r == 1.0:
                    return 1.0
                if r == 0:
                    return 1.0
            if s_p == s and s % self.grid_world.columns == 0:
                if r == 0:
                    return 1.0

        if a == 1:
            if s % self.grid_world.columns != self.grid_world.rows - 1 and s_p == s + 1:
                if s_p == self.grid_world.lose_pos and r == -1:
                    return 1.0
                if s_p == self.grid_world.win_pos and r == 1:
                    return 1.0
                if r == 0:
                    return 1.0
            if s_p == s and s % self.grid_world.columns == self.grid_world.columns - 1:
                if r == 0:
                    return 1.0

        if a == 2:
            if s >= self.grid_world.columns and s_p == s - self.grid_world.columns:

                if s_p == self.grid_world.lose_pos and r == -1:
                    return 1.0
                if s_p == self.grid_world.win_pos and r == 1:
                    return 1.0
                if r == 0:
                    return 1.0
            if s_p == s and s < self.grid_world.columns:
                if r == 0:
                    return 1.0

        if a == 3:
            if s <= self.grid_world.win_pos - self.grid_world.columns and s_p == s + self.grid_world.columns:
                if s_p == self.grid_world.lose_pos and r == -1:
                    return 1.0
                if s_p == self.grid_world.win_pos and r == 1:
                    return 1.0
                if r == 0:
                    return 1.0
            if s_p == s and s > (self.grid_world.columns * self.grid_world.rows - 1):
                if r == 0:
                    return 1.0

        return 0.0

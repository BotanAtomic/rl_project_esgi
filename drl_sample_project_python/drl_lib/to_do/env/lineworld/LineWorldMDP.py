import numpy as np

from drl_lib.do_not_touch.contracts import MDPEnv


class LineWorldMDP(MDPEnv):
    def __init__(self, cell_count: int):
        assert (cell_count >= 3)
        self.cell_count = cell_count
        self.S = np.arange(cell_count)
        self.A = np.array([0, 1])
        self.R = np.array([-1, 0, 1])
        self.p = self.get_transition_matrix()

    def get_transition_matrix(self) -> np.ndarray:
        p = np.zeros((len(self.S), len(self.A), len(self.S), len(self.R)))

        for i in range(1, self.cell_count - 2):
            p[i, 1, i + 1, 1] = 1.0

        for i in range(2, self.cell_count - 1):
            p[i, 0, i - 1, 1] = 1.0

        p[self.cell_count - 2, 1, self.cell_count - 1, 2] = 1.0
        p[1, 0, 0, 0] = 1.0

        return p

    def states(self) -> np.ndarray:
        return self.S

    def actions(self) -> np.ndarray:
        return self.A

    def rewards(self) -> np.ndarray:
        return self.R

    def is_state_terminal(self, s: int) -> bool:
        return s == 0 or (s == self.cell_count - 1)

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self.p[s, a, s_p, r]

    def view_state(self, s: int):
        pass

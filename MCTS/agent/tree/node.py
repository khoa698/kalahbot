from MCTS.environment.mancala import MancalaEnv
from MCTS.environment.move import Move
from numpy.ma import sqrt
from copy import deepcopy


class Node:
    def __init__(self, state: MancalaEnv, p: float = 0.5, move: Move = None, parent=None):
        self.parent = parent
        self.state = state
        self.children = []
        self.unexplored_moves = set(state.get_legal_moves())
        self.visits = 0
        self.reward = 0
        self.value = -1
        self.move = move
        self.p = p
        self.explore = p / (1 + self.visits)

    @staticmethod
    def clone(other_node):
        return deepcopy(other_node)

    def update(self, reward: float, constant: int = 3):
        self.visits += 1
        self.reward += ((reward - self.reward) / self.visits)
        if self.parent is not None:
            self.explore = constant * self.p * sqrt(self.parent.visits) / (1 + self.visits)

    def put_child(self, child):
        self.children.append(child)
        self.unexplored_moves.remove(child.move)

    def is_leaf_node(self) -> bool:
        return len(self.state.get_legal_moves()) == 0

    def is_explored(self) -> bool:
        return len(self.unexplored_moves) == 0

    def action_val(self) -> float:
        return self.reward + self.explore

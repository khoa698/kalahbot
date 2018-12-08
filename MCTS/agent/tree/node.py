from copy import deepcopy
from MCTS.environment.mancala import MancalaEnv
from MCTS.environment.move import Move


class Node:
    def __init__(self, state: MancalaEnv, move: Move = None, parent=None):
        self.parent = parent
        self.state = state
        self.children = []
        self.unexplored_moves = set(state.get_legal_moves())
        self.visits = 0
        self.reward = 0
        self.move = move

    @staticmethod
    def clone(other_node):
        return deepcopy(other_node)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def put_child(self, child):
        self.children.append(child)
        self.unexplored_moves.remove(child.move)

    def is_leaf_node(self) -> bool:
        return len(self.state.get_legal_moves()) == 0

    def is_explored(self) -> bool:
        return len(self.unexplored_moves) == 0

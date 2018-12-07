from env.mancala import MancalaEnv
from env.move import Move
from copy import deepcopy

class Node:

    def __init__(self, state: MancalaEnv, parent=None, move: Move = None):
        self.state = state
        self.parent = parent
        self.reward = 0
        self.visits = 0
        self.explored_children = []
        self.move = move
        self.unexplored_moves = set(state.get_legal_moves())

    @staticmethod
    def clone(self):
        return deepcopy(self)

    def append_explored_child(self, child):

        if child in self.unexplored_moves:
            self.explored_children.append(child)
        else:
            raise Exception("Invalid child node")

    def update_node(self, reward: int):
        self.reward += reward
        self.visits = self.visits + 1

    def is_leaf_node(self)->bool:
        return len(self.state.get_legal_moves()) == 0

    def get_legal_moves(self):
        return self.get_legal_moves()

    def is_fully_expanded(self) -> bool:
        return len(self.unexplored_moves) == 0












from mcts.node import Node
import mcts.utils as utils
from random import randint
from env.mancala import MancalaEnv
from env.side import Side


class TreePolicy:
    def select(self, node: Node)->Node:
        current_node = node
        while not current_node.is_leaf_node() and not current_node.value == -1:
            if not len(current_node.explored_children) == 0:
                return self.expand(current_node)
            else:
                current_node = utils.select_best_move(current_node)
        return current_node

    @staticmethod
    def expand(node: Node)->Node:
        index = randint(0, len(node.unexplored_moves) - 1)
        new_state = MancalaEnv.clone(node.state)
        new_state.make_move(new_state.board, node.unexplored_moves[index], node.state.side_to_move)
        new_node = Node(new_state, node, node.unexplored_moves[index])
        return new_node







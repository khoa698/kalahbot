from mcts.node import Node
import mcts.utils as utils
from random import randint
from env.mancala import MancalaEnv
from env.side import Side


def select(node: Node)->Node:
    current_node = node
    while not current_node.is_leaf_node():
        if not len(current_node.explored_children) == 0:
            return expand(current_node)
        else:
            current_node = utils.select_best_child(current_node)
    return current_node

def expand(node: Node)->Node:
    index = randint(0, len(node.unexplored_moves) - 1)
    new_state = MancalaEnv.clone(node.state)
    move = node.unexplored_moves[index]
    node.unexplored_moves.remove(move)
    new_state.make_move(new_state.board, move, node.state.side_to_move)
    new_node = Node(new_state, node, move)
    return new_node







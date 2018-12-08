from mcts.node import Node
from env.mancala import MancalaEnv
from env.side import Side
from env.move import Move

import math


def select_best_move(node: Node)->Move:
    if node.is_leaf_node():
        raise Exception("No children")
    else:
        best_child = None
        max_value = 0
        for child in node.explored_children:
            if max_value < compute_uct(node, child):
                max_value = max_value
                best_child = child
        return best_child.move


def select_best_child(node: Node) -> Node:
    if node.is_leaf_node():
        raise Exception("No children")
    else:
        best_child = None
        max_value = 0
        for child in node.explored_children:
            if max_value < compute_uct(node, child):
                max_value = max_value
                best_child = child

        return best_child


def compute_uct(parent: Node, child: Node, exploration_constant: float = math.sqrt(2))->float:
    return (child.reward / child.visits) - (exploration_constant * math.sqrt(2 * math.log(parent.visits / child.visits)))


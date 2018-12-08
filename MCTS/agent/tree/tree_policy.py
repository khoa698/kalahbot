from random import choice

from MCTS.environment.mancala import MancalaEnv
from MCTS.agent.tree import utilities
from MCTS.agent.tree.node import Node


def select(node: Node) -> Node:
    while not node.is_leaf_node():
        if not node.is_explored():
            return expand(node)
        else:
            node = utilities.select_child(node)
    return node


def expand(parent: Node) -> Node:
    child_expansion_move = choice(tuple(parent.unexplored_moves))
    child_state = MancalaEnv.clone(parent.state)
    child_state.perform_move(child_expansion_move)
    child_node = Node(state=child_state, move=child_expansion_move, parent=parent)
    parent.put_child(child_node)
    return child_node





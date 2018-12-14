from MCTS.agent.tree.node import Node
from math import log, sqrt

from MCTS.environment.mancala import MancalaEnv
from MCTS.environment.side import Side


def uct(parent: Node, child: Node, ec: float = 1 / sqrt(2)) -> float:
    return (child.reward / child.visits) + (ec * sqrt(2 * log(parent.visits) / child.visits))


def select_child(node: Node) -> Node:
    if node.is_leaf_node():
        raise ValueError('Leaf node reached. No children')
    elif len(node.children) == 1:
        return node.children[0]

    return max(node.children, key=lambda child: uct(node, child))


def h1(state: MancalaEnv, parent_side: Side) -> int:
    my_mancala = state.board.get_seeds_in_store(parent_side)
    opponent_mancala = state.board.get_seeds_in_store(Side.opposite(parent_side))

    diff = my_mancala - opponent_mancala

    return diff





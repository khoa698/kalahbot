from MCTS.agent.tree.node import Node
from math import log, sqrt


def select_child(node: Node) -> Node:
    return max(node.children, key=lambda child: child.action_val())





from mcts.node import Node


def backpropagate(leaf: Node):
    winner = leaf.state.get_winner()
    current_node = leaf
    while current_node is not None and current_node.parent is not None:
        if current_node.state.get_winner() == winner:
            current_node.update_node(1)
        current_node.update_node(0)
        parent_node = current_node.parent
        parent_node.append_explored_child(current_node)
        current_node = parent_node




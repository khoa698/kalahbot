from mcts.node import Node
from random import randint
from env.mancala import MancalaEnv


class DefaultPolicy:

    @staticmethod
    def simulate(node: Node)->Node:
        current_node = node
        parent_node = node
        while not current_node.is_leaf_node():
            index = randint(0, len(current_node.unexplored_moves) - 1)
            new_state = MancalaEnv.clone(current_node.state)
            new_state.make_move(new_state.board, current_node.unexplored_moves[index],
                                current_node.state.side_to_move)
            current_node = Node(new_state, parent_node, current_node.unexplored_moves[index])
            parent_node = current_node
        return current_node

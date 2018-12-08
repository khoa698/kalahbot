from MCTS.environment.move import Move
from MCTS.environment.mancala import MancalaEnv
from MCTS.agent.tree.node import Node
import numpy as np


def simulate(root: Node) -> MancalaEnv:
    node = Node.clone(root)
    while not node.is_leaf_node():
        legal_moves = node.state.get_legal_moves()
        move_indices = []
        for move in legal_moves:
            move_indices.append(move.index)

        next_move = int(np.random.choice(move_indices))
        node.state.perform_move(Move(side=node.state.side_to_move, index=next_move))

    return node.state


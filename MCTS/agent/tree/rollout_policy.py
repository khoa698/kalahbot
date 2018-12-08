from MCTS.agent.tree.node import Node
from MCTS.environment.mancala import MancalaEnv


def backpropagate(root: Node, final_state: MancalaEnv):
    node = root
    while node is not None:
        side = node.parent.state.side_to_move if node.parent is not None else node.state.side_to_move  # root node
        node.update(final_state.compute_end_game_reward(side))
        node = node.parent


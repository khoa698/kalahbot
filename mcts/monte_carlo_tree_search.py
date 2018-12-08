from mcts.node import Node
from env.move import Move
from mcts.default_policy import simulate
from mcts.tree_policy import select
from mcts.rollout_policy import backpropagate
import mcts.utils as utils
import datetime

class MonteCarloTreeSearch:
    def __init__(self, duration: int):
        self.duration = duration

    def find_next_move(self, node: Node)->Move:
        print(len(node.get_legal_moves()))
        if len(node.get_legal_moves()) == 1:
            return node.get_legal_moves()[0]

        starting_time = datetime.datetime.utcnow()
        current_node = node
        while datetime.datetime.utcnow() - starting_time < datetime.timedelta(seconds=self.duration):
            current_node = select(node)
            leaf_node = simulate(current_node)
            backpropagate(leaf_node)
        return utils.select_best_move(current_node)






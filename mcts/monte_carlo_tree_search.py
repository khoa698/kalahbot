from mcts.node import Node
from env.move import Move
from mcts.default_policy import DefaultPolicy
from mcts.tree_policy import TreePolicy
from mcts.rollout_policy import RolloutPolicy
import mcts.utils as utils
import datetime

class MonteCarloTreeSearch:
    def __init__(self, root: Node, duration: int, tree_policy: TreePolicy,
                 default_policy: DefaultPolicy, rollout_policy: RolloutPolicy):
        self.root = root
        self.duration = duration
        self.tree_policy = tree_policy
        self.default_policy = default_policy
        self.rollout_policy = rollout_policy

    def findNextMove(self)->Move:

        if len(self.root.get_legal_moves()) == 1:
            return self.root.get_legal_moves()[0]

        starting_time = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - starting_time < datetime.timedelta(seconds=self.duration):
            current_node = self.tree_policy.select(self.root)
            leaf_node = self.default_policy.simulate(current_node)
            self.rollout_policy.backpropagate(leaf_node)
            return utils.select_best_move(current_node)






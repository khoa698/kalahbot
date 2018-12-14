import datetime

from MCTS.environment.kalah import KalahEnvironment
from MCTS.environment.move import Move
from MCTS.agent.tree import utilities
from MCTS.agent.tree.node import Node
from MCTS.agent.tree.policies import MonteCarloA3CPolicies


class MCTS:
    def __init__(self, run_duration: int, policies: MonteCarloA3CPolicies):
        self.run_duration: datetime.timedelta = datetime.timedelta(seconds=run_duration)
        self.policies = policies

    def find_next_move(self, state: KalahEnvironment) -> Move:
        if len(state.get_valid_moves()) == 1:
            return state.get_valid_moves()[0]

        root = Node(state=KalahEnvironment.clone(state))
        time = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - time < self.run_duration:
            node = self.policies.select(root)
            ending_state = self.policies.simulate(node)
            self.policies.backpropagate(node, ending_state)
        chosen = utilities.select_child(root)
        return chosen.move

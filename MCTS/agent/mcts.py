import datetime

from MCTS.environment.mancala import MancalaEnv
from MCTS.environment.move import Move
from MCTS.agent.tree import utilities
from MCTS.agent.tree.node import Node
from MCTS.agent.tree.default_policy import simulate
from MCTS.agent.tree.rollout_policy import backpropagate
from MCTS.agent.tree.tree_policy import select


class MCTS:
    def __init__(self, run_duration: int):
        self.run_duration: datetime.timedelta = datetime.timedelta(seconds=run_duration)

    def find_next_move(self, state: MancalaEnv) -> Move:
        if len(state.get_legal_moves()) == 1:
            return state.get_legal_moves()[0]

        root = Node(state=MancalaEnv.clone(state))
        start_time = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - start_time < self.run_duration:
            node = select(root)
            ending_state = simulate(node)
            backpropagate(node, ending_state)
        chosen = utilities.select_child(root)
        return chosen.move

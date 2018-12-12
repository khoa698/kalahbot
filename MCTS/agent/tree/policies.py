from MCTS.environment.move import Move
from MCTS.environment.mancala import MancalaEnv
from MCTS.agent.tree import utilities
from MCTS.agent.tree.node import Node
from MCTS.agent.a3c.a3client import A3Client


class MonteCarloA3CPolicies:
    def __init__(self, a3c_network: A3Client):
        self.a3c_network = a3c_network

    def simulate(self, root: Node) -> MancalaEnv:
        node = Node.clone(root)
        while not node.is_leaf_node():
            move_index, _ = self.a3c_network.sample(node.state)
            move = Move(node.state.side_to_move, move_index + 1)
            node.state.perform_move(move)
        return node.state

    def backpropagate(self, root: Node, final_state: MancalaEnv):
        stack = []
        node = root
        while node is not None:
            stack.append(node)
            node = node.parent
        while len(stack) > 0:
            node = stack.pop()
            side = node.parent.state.side_to_move if node.parent is not None else node.state.side_to_move
            reward = final_state.compute_end_game_reward(side)
            node.update(reward)

    def select(self, node: Node) -> Node:
        while not node.is_leaf_node():
            if not node.is_explored():
                return self.expand(node)
            else:
                node = utilities.select_child(node)
        return node

    def expand(self, parent: Node) -> Node:
        if Move(parent.state.side_to_move, 0) in parent.unexplored_moves:
            parent.unexplored_moves.remove(Move(parent.state.side_to_move, 0))

        dist, value = self.a3c_network.evaluate(parent.state)
        for index, p_val in enumerate(dist):
            move = Move(parent.state.side_to_move, index + 1)
            if parent.state.is_legal(move):
                child_state = MancalaEnv.clone(parent.state)
                child_state.perform_move(move)
                child_node = Node(state=child_state, p=p_val, move=move, parent=parent)
                parent.put_child(child_node)
        return utilities.select_child(parent)





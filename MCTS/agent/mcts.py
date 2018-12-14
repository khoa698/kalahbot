import datetime
import multiprocessing
from multiprocessing import Process, Queue
from MCTS.environment.kalah import KalahEnvironment
from MCTS.environment.move import Move
from MCTS.agent.tree import utilities
from MCTS.agent.tree.node import Node
from MCTS.agent.tree.policies import MonteCarloA3CPolicies

import logging
logging.basicConfig(filename='log.log', level=logging.DEBUG)

class MCTS:
    def __init__(self, run_duration: int, policies: MonteCarloA3CPolicies):
        self.run_duration: datetime.timedelta = datetime.timedelta(seconds=run_duration)
        self.policies = policies

    def find_next_move(self, state: KalahEnvironment, output_queue):
        if len(state.get_valid_moves()) == 1:
            return state.get_valid_moves()[0]

        root = Node(state=KalahEnvironment.clone(state))
        time = datetime.datetime.utcnow()

        while datetime.datetime.utcnow() - time < self.run_duration:
            node = self.policies.select(root)
            ending_state = self.policies.simulate(node)
            self.policies.backpropagate(node, ending_state)
        chosen = utilities.select_child(root)
        output_queue.put([chosen.move, chosen.reward])

    def parrallelized(self, state: KalahEnvironment) -> Move:
        process_list = []
        ensemble_count = 8
        output_queue = Queue(ensemble_count)
        for proc in range(ensemble_count):
            worker_proc = Process(target=self.find_next_move, args=(state, output_queue))
            worker_proc.daemon = True
            process_list.append(worker_proc)
            worker_proc.start()
        for worker in process_list:
            worker.join()

        options = []
        rewards = []
        for _ in range(ensemble_count):
            output = output_queue.get()
            options.append(output[0])
            rewards.append(output[1])
        logging.info(options)
        logging.info(rewards)

        return options[rewards.index(max(rewards))]

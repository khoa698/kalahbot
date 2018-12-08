import datetime
import logging
import traceback
import tensorflow as tf

import protocol.protocol as protocol
from a3c.env.mancala import MancalaEnv
from a3c.env.move import Move
from protocol.invalid_message_exception import InvalidMessageException
from protocol.msg_type import MsgType
from a3c.env.side import Side
from mcts.node import Node
from mcts.monte_carlo_tree_search import MonteCarloTreeSearch
# from magent.treesearch.factory import TreesFactory
# from models.client import A3Client

# set up logging to file - see previous section for more details

def main(_):
    """with tf.Session() as sess:
        with tf.variable_scope("global"):
            a3client = A3Client(sess)
            mcts = TreesFactory.alpha_mcts(a3client)
            state = MancalaEnv()
            try:
                _run_game(mcts, state)
            except Exception as e:
                logging.error("Uncaught exception in main: " + str(e))
                # TODO uncomment before release: Default to reasonable move behaviour on failure
                # protocol.send_msg(protocol.create_move_msg(choice(state.get_legal_moves())))"""

    state = MancalaEnv()
    root = Node(state)

    mcts = MonteCarloTreeSearch(root=root,duration=1000000000000)
    try:
        _run_game(mcts, state)
    except Exception as e:
        logging.error("Uncaught exception in main: " + str(e))
        # TODO uncomment before release: Default to reasonable move behaviour on failure
        # protocol.send_msg(protocol.create_move_msg(choice(state.get_legal_moves())))


def _run_game(mcts, state):
    while True:
        msg = protocol.read_msg()
        try:
            msg_type = protocol.get_msg_type(msg)
            if msg_type == MsgType.START:
                first = protocol.interpret_start_msg(msg)
                if first:
                    node = Node(state)

                    move = mcts.find_next_move(node)

                    protocol.send_msg(protocol.create_move_msg(move.index))
                else:
                    state.our_side = Side.NORTH
            elif msg_type == MsgType.STATE:
                move_turn = protocol.interpret_state_msg(msg)
                state.perform_move(Move(state.side_to_move, move_turn.move))
                if not move_turn.end:
                    if move_turn.again:
                        node = Node(state)
                        move = mcts.find_next_move(node)
                        # pie rule; optimal move is to swap
                        if move.index == 0:
                            protocol.send_msg(protocol.create_swap_msg())
                        else:
                            protocol.send_msg(protocol.create_move_msg(move.index))

                logging.info("Next side: " + str(state.side_to_move))
                logging.info("The board:\n" + str(state.board))
            elif msg_type == MsgType.END:
                break
            else:
                logging.warning("Not sure what I got " + str(msg_type))
        except InvalidMessageException as e:
            logging.error(str(e))


def mcts_main():
    state = MancalaEnv()
    root = Node(state)

    mcts = MonteCarloTreeSearch(duration=10)
    try:
        _run_game(mcts, state)
    except Exception as e:
        logging.error("Uncaught exception in main: " + str(e))
        traceback.print_exc()
        # TODO uncomment before release: Default to reasonable move behaviour on failure
        # protocol.send_msg(protocol.create_move_msg(choice(state.get_legal_moves())))


if __name__ == '__main__':
    # tf.app.run()
    mcts_main()
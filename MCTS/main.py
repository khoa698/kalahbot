from MCTS.agent.mcts import MCTS
from MCTS.agent.tree.policies import MonteCarloA3CPolicies
from MCTS.agent.a3c.a3client import A3Client
import MCTS.protocol.protocol as protocol
from MCTS.environment.move import Move
from MCTS.protocol.msg_type import MsgType
from MCTS.environment.mancala import MancalaEnv
from MCTS.environment.side import Side
import tensorflow as tf

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def run(mcts, state):
    while True:
        msg = protocol.read_msg()
        msg_type = protocol.get_msg_type(msg)
        if msg_type == MsgType.START:
            first = protocol.interpret_start_msg(msg)
            if first:
                move = mcts.find_next_move(state)
                protocol.send_msg(protocol.create_move_msg(move.index))
            else:
                state.our_side = Side.NORTH
        elif msg_type == MsgType.STATE:
            move_turn = protocol.interpret_state_msg(msg)
            state.perform_move(Move(state.side_to_move, move_turn.move))
            if not move_turn.end:
                if move_turn.again:
                    move = mcts.find_next_move(state)
                    if move.index == 0:
                        protocol.send_msg(protocol.create_swap_msg())
                    else:
                        protocol.send_msg(protocol.create_move_msg(move.index))

        else:
            break


def main():
    with tf.Session(config=tf.ConfigProto(
            intra_op_parallelism_threads=12)) as sess:
        with tf.variable_scope("global"):
            a3c_network = A3Client(sess)
            mcts = MCTS(run_duration=20, policies=MonteCarloA3CPolicies(a3c_network=a3c_network))
            state = MancalaEnv()
            run(mcts, state)


if __name__ == '__main__':
    main()


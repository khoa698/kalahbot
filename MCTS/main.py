from MCTS.agent.mcts import MCTS
import MCTS.protocol.protocol as protocol
from MCTS.agent.tree.policies import MonteCarloA3CPolicies
from MCTS.environment.move import Move
from MCTS.protocol.msg_type import MsgType
from MCTS.environment.kalah import KalahEnvironment
from MCTS.environment.side import Side


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
                state.my_side = Side.NORTH
        elif msg_type == MsgType.STATE:
            move_turn = protocol.interpret_state_msg(msg)
            state.do_move(Move(state.side_to_play, move_turn.move))
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
    policies = MonteCarloA3CPolicies()
    mcts = MCTS(run_duration=20, policies=policies)
    state = KalahEnvironment()
    run(mcts, state)


if __name__ == '__main__':
    main()


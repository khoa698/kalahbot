from MCTS.environment.mancala import MancalaEnv
from MCTS.environment.side import Side


def h1(state: MancalaEnv, side: Side) -> float:
    my_mancala = state.board.get_seeds_in_store(side)
    opponent_mancala = state.board.get_seeds_in_store(Side.opposite(side))

    diff = my_mancala - opponent_mancala

    return diff


def h2(state: MancalaEnv, side: Side) -> float:
    current_score = state.board.get_seeds_in_store(side)
    if current_score > 49:
        return current_score-50
    return 100/(50-current_score)


def get_heuristics(state: MancalaEnv, side: Side) -> float:
    heuristics_val = h1(state, side)
    heuristics_val *= h2(state, side)
    return heuristics_val
from MCTS.environment.board import Board
from MCTS.environment.side import Side
from MCTS.environment.move import Move
from copy import deepcopy
from typing import List
import numpy as np


class KalahEnvironment(object):

    def __init__(self):
        self.board = Board(7, 7)
        self.side_to_play = Side.SOUTH
        self.my_side = Side.SOUTH
        self.north_has_moved = False

    def reset(self):
        self.board = Board(7, 7)
        self.side_to_play = Side.SOUTH
        self.my_side = Side.SOUTH
        self.north_has_moved = False

    @property
    def my_side(self):
        return self._my_side

    @property
    def board(self):
        return self._board

    @board.setter
    def board(self, board: Board):
        self._board = board

    @my_side.setter
    def my_side(self, side: Side):
        self._my_side = side

    @property
    def side_to_play(self):
        return self._side_to_move

    @side_to_play.setter
    def side_to_play(self, side: Side):
        self._side_to_move = side

    @property
    def north_has_moved(self):
        return self._north_moved

    @north_has_moved.setter
    def north_has_moved(self, moved: bool):
        self._north_moved = moved

    @staticmethod
    def clone(other):
        board = Board.clone(other.board)
        north_moved = deepcopy(other.north_has_moved)
        side_to_move = deepcopy(other.side_to_play)
        cloned_env = KalahEnvironment()
        cloned_env.north_has_moved = north_moved
        cloned_env.side_to_play = side_to_move
        cloned_env.board = board
        return cloned_env

    def do_move(self, move: Move):
        if move.index == 0:
            self.my_side = Side.opposite(self.my_side)
        self.side_to_play = KalahEnvironment.update_env_after_move(self.board, move, self.north_has_moved)
        if move.side == Side.NORTH:
            self.north_has_moved = True

    def get_valid_moves(self) -> List[Move]:
        return KalahEnvironment.get_valid_moves_at_state(self.board, self.side_to_play, self.north_has_moved)

    def get_reward_for_winning(self, side: Side)->float:
        if not self.has_game_ended():
            raise Exception('Game has not ended')
        reward = self.calculate_score_diff(side)
        if reward > 0:
            return 1
        elif reward < 0:
            return 0
        else:
            return 0.5

    def calculate_score_diff(self, side: Side):
        diff = self.board.get_seeds_in_store(side) - self.board.get_seeds_in_store(Side.opposite(side))
        return diff

    def get_winner(self) -> Side or None:
        if not self.has_game_ended():
            raise Exception('Game has not ended')
        last_move_side = Side.NORTH if KalahEnvironment.side_has_no_seeds(self.board, Side.NORTH) else Side.SOUTH
        other_side = Side.opposite(last_move_side)
        last_move_side_seeds = self.board.get_seeds_in_store(other_side)
        for hole in range(1, self.board.holes + 1):
            last_move_side_seeds += self.board.get_seeds(other_side, hole)
        other_side_seeds = self.board.get_seeds_in_store(last_move_side)
        if other_side_seeds > last_move_side_seeds:
            return last_move_side
        elif other_side_seeds < last_move_side_seeds:
            return other_side
        else:
            return None

    def has_game_ended(self) -> bool:
        return KalahEnvironment.game_finished(self.board)

    def get_mask(self) -> [float]:
        mask = [0 for _ in range(self.board.holes)]
        moves = [move.index for move in self.get_valid_moves()]
        if 0 in moves:
            moves.remove(0)
        for action in moves:
            mask[action - 1] = 1
        return np.array(mask)

    @staticmethod
    def is_permitted(board: Board, move: Move, north_moved: bool) -> bool:
        return move.index in [action.index for action in KalahEnvironment.get_valid_moves_at_state(board, move.side, north_moved)]

    @staticmethod
    def side_has_no_seeds(board: Board, side: Side):
        for hole in range(1, board.holes + 1):
            if board.get_seeds(side, hole) > 0:
                return False
        return True

    @staticmethod
    def get_valid_moves_at_state(board: Board, side: Side, north_moved: bool) -> List[Move]:
        valid_moves = [] if north_moved or side == side.SOUTH else [Move(side, 0)]
        for hole in range(1, board.holes + 1):
            if board.board[side.get_index(side)][hole] > 0:
                valid_moves.append(Move(side, hole))
        return valid_moves

    @staticmethod
    def update_env_after_move(board: Board, move: Move, north_moved):
        if not KalahEnvironment.is_permitted(board, move, north_moved):
            raise Exception('Move not permitted')
        if move.index == 0:
            KalahEnvironment.swap_sides(board)
            return Side.opposite(move.side)
        seeds_to_sow = board.get_seeds(move.side, move.index)
        board.set_seeds(move.side, move.index, 0)
        holes = board.holes
        receiving_holes = 2 * holes + 1
        rounds = seeds_to_sow // receiving_holes
        remaining_seeds = seeds_to_sow % receiving_holes
        if rounds != 0:
            for hole in range(1, holes + 1):
                board.add_seeds(Side.NORTH, hole, rounds)
                board.add_seeds(Side.SOUTH, hole, rounds)
            board.add_seeds_to_store(move.side, rounds)
        sow_side = move.side
        sow_hole = move.index
        for _ in range(remaining_seeds):
            sow_hole += 1
            if sow_hole == 1:
                sow_side = Side.opposite(sow_side)
            if sow_hole > holes:
                if sow_side == move.side:
                    sow_hole = 0
                    board.add_seeds_to_store(sow_side, 1)
                    continue
                else:
                    sow_side = Side.opposite(sow_side)
                    sow_hole = 1
            board.add_seeds(sow_side, sow_hole, 1)
        if sow_side == move.side and sow_hole > 0 and board.get_seeds(sow_side, sow_hole) == 1 \
                and board.get_seeds_op(sow_side, sow_hole) > 0:
            board.add_seeds_to_store(move.side, 1 + board.get_seeds_op(sow_side, sow_hole))
            board.set_seeds(move.side, sow_hole, 0)
            board.set_seeds_op(move.side, sow_hole, 0)
        game_over = KalahEnvironment.game_finished(board)
        if game_over:
            finished_side = Side.NORTH if KalahEnvironment.side_has_no_seeds(board, Side.NORTH) else Side.SOUTH
            seeds = 0
            collecting_side = Side.opposite(finished_side)
            for hole in range(1, board.holes + 1):
                seeds += board.get_seeds(collecting_side, hole)
                board.set_seeds(collecting_side, hole, 0)
            board.add_seeds_to_store(collecting_side, seeds)
        if sow_hole == 0 and (move.side == Side.NORTH or north_moved):
            return move.side
        return Side.opposite(move.side)

    @staticmethod
    def swap_sides(board: Board):
        for i in range(board.holes + 1):
            board.board[0][i], board.board[1][i] = board.board[1][i], board.board[0][i]

    @staticmethod
    def game_finished(board: Board):
        if KalahEnvironment.side_has_no_seeds(board, Side.SOUTH):
            return True
        if KalahEnvironment.side_has_no_seeds(board, Side.NORTH):
            return True
        return False

    def __str__(self):
        return "%s" % self.board


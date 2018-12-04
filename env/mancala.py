from typing import List
from copy import deepcopy

import numpy as np

from env.board import Board
from env.move import Move
from env.side import Side


class MancalaEnv(object):

    def __init__(self):
        self.board = Board(7, 7)
        self.side_to_move = Side.SOUTH
        self.north_moved = False
        self.my_side = Side.SOUTH

    def reset(self):
        self.board = Board(7, 7)
        self.side_to_move = Side.SOUTH
        self.north_moved = False
        self.my_side = Side.SOUTH

    def get_board(self):
        return self.board

    def get_side_to_move(self):
        return self.side_to_move

    def set_side_to_move(self, side: Side):
        self.side_to_move = side

    def set_north_moved(self, moved: bool):
        self.north_moved = moved

    def get_my_side(self):
        return self.my_side

    def get_legal_moves(self) -> List[Move]:
        return MancalaEnv.get_state_legal_actions(self.board, self.side_to_move, self.north_moved)

    def is_legal(self, move: Move) -> bool:
        return MancalaEnv.is_legal_action(self.board, move, self.north_moved)

    def perform_move(self, move: Move) -> int:
        num_of_store_seeds_before = self.board.get_store(move.side)
        if move.side == 0:
            self.my_side = Side.opposite(self.my_side)
        self.side_to_move = MancalaEnv.make_move(self.board, move, self.north_moved)
        if move.side == Side.NORTH:
            self.north_moved = True
        num_of_store_seeds_after = self.board.get_store(move.side)
        return (num_of_store_seeds_after - num_of_store_seeds_before) / 100.0

    def compute_final_reward(self, side: Side):
        reward = self.board.get_store(side) - self.board.get_store(Side.opposite(side))
        return reward

    def is_game_over(self) -> bool:
        return MancalaEnv.game_over(self.board)

    def get_action_mask_with_no_pie(self) -> [float]:
        mask = [0 for _ in range(self.board.holes)]
        moves = [move.hole for move in self.get_legal_moves()]
        if 0 in moves:
            moves.remove(0)
        for action in moves:
            mask[action - 1] = 1
        return np.array(mask)

    def get_winner(self) -> Side or None:
        if not self.is_game_over():
            raise Exception('Game is not finished')

        if MancalaEnv.holes_empty(self.board, Side.NORTH):
            finished_side = Side.NORTH
        else:
            finished_side = Side.SOUTH

        not_finished_side = Side.opposite(finished_side)
        not_finished_side_seeds = self.board.get_store(not_finished_side)
        for hole in range(1, self.board.holes + 1):
            not_finished_side_seeds += self.board.get_seeds_in_hole(not_finished_side, hole)
        finished_side_seeds = self.board.get_store(finished_side)
        if finished_side_seeds > not_finished_side_seeds:
            return finished_side
        elif finished_side_seeds < not_finished_side_seeds:
            return not_finished_side
        return None

    @staticmethod
    def get_state_legal_actions(board: Board, side: Side, north_moved: bool) -> List[Move]:
        legal_moves = [] if north_moved or side == side.SOUTH else [Move(side, 0)]
        for i in range(1, board.holes + 1):
            if board.board[side.get_side(side)][i] > 0:
                legal_moves.append(Move(side, i))
        return legal_moves

    @staticmethod
    def is_legal_action(board: Board, move: Move, north_moved: bool) -> bool:
        return move.hole in [act.hole for act in MancalaEnv.get_state_legal_actions(board, move.side, north_moved)]

    @staticmethod
    def holes_empty(board: Board, side: Side):
        for hole in range(1, board.holes + 1):
            if board.get_seeds_in_hole(side, hole) > 0:
                return False
        return True

    @staticmethod
    def game_over(board: Board):
        if MancalaEnv.holes_empty(board, Side.SOUTH):
            return True
        if MancalaEnv.holes_empty(board, Side.NORTH):
            return True
        return False

    @staticmethod
    def switch_sides(board: Board):
        for hole in range(board.holes + 1):
            board.board[0][hole], board.board[1][hole] = board.board[1][hole], board.board[0][hole]

    @staticmethod
    def make_move(board: Board, move: Move, north_moved):
        if not MancalaEnv.is_legal_action(board, move, north_moved):
            raise Exception('illegal move')

        if move.hole == 0:
            MancalaEnv.switch_sides(board)
            return Side.opposite(move.side)

        seeds = board.get_seeds_in_hole(move.side, move.hole)
        board.set_seeds_in_hole(move.side, move.hole, 0)

        holes = board.holes
        receiving_holes = 2 * holes + 1
        rounds = seeds
        remaining_seeds = seeds % receiving_holes

        if rounds != 0:
            for hole in range(1, holes + 1):
                board.add_seeds_in_hole(Side.NORTH, hole, rounds)
                board.add_seeds_in_hole(Side.SOUTH, hole, rounds)
            board.add_to_store(move.side, rounds)

        side = move.side
        hole = move.hole
        for _ in range(remaining_seeds):
            hole += 1
            if hole == 1:
                side = Side.opposite(side)
            if hole > holes:
                if side == move.side:
                    hole = 0
                    board.add_to_store(side, 1)
                    continue
                else:
                    side = Side.opposite(side)
                    hole = 1
            board.add_seeds_in_hole(side, hole, 1)
        if side == move.side and hole > 0 and board.get_seeds_in_hole(side, hole) == 1 \
                and board.get_seeds_op(side, hole) > 0:
            board.add_to_store(move.side, 1 + board.get_seeds_op(side, hole))
            board.set_seeds_in_hole(move.side, hole, 0)
            board.set_seeds_op(move.side, hole, 0)

        game_over = MancalaEnv.game_over(board)
        if game_over:
            finished_side = Side.NORTH if MancalaEnv.holes_empty(board, Side.NORTH) else Side.SOUTH
            seeds = 0
            collecting_side = Side.opposite(finished_side)
            for hole in range(1, board.holes + 1):
                seeds += board.get_seeds_in_hole(collecting_side, hole)
                board.set_seeds_in_hole(collecting_side, hole, 0)
            board.add_to_store(collecting_side, seeds)

        if hole == 0 and (move.side == Side.NORTH or north_moved):
            return move.side
        return Side.opposite(move.side)




















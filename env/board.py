from copy import deepcopy
from env.side import Side
import numpy as np


class Board(object):

    def __init__(self, holes: int, seeds: int):
        if holes < 1:
            raise Exception("Need at least 1 hole")
        if seeds < 0:
            raise Exception("Need at least 1 seed")

        self.holes = holes
        self.board = [[0 for _ in range(holes + 1)] for _ in range(2)]
        for hole in range(1, holes + 1):
            self.board[Side.get_side(Side.NORTH)][hole] = seeds
            self.board[Side.get_side(Side.SOUTH)][hole] = seeds

    @classmethod
    def clone(cls, original_board):
        holes = original_board.holes
        board = cls(holes, 0)
        for hole in range(1, holes + 1):
            board.board[Side.get_side(Side.NORTH)][hole] = deepcopy(original_board.board[Side.get_side(Side.NORTH)][hole])
            board.board[Side.get_side(Side.SOUTH)][hole] = deepcopy(original_board.board[Side.get_side(Side.SOUTH)][hole])
        return board

    def get_seeds_in_hole(self, side: Side, hole: int)->int:
        if hole < 1 or hole > self.holes:
            raise Exception('Invalid hole number')
        return self.board[Side.get_side(side)][hole]

    def set_seeds_in_hole(self, side: Side, hole: int, seeds: int):
        if hole < 1 or hole > self.holes:
            raise Exception('Invalid hole number')
        if seeds < 0:
            raise Exception('seeds number is negative')
        self.board[Side.get_side(side)][hole] = seeds

    def add_seeds_in_hole(self, side: Side, hole: int, seeds: int):
        if hole < 1 or hole > self.holes:
            raise Exception('Invalid hole number')
        if seeds < 0:
            raise Exception('seeds number is negative')
        self.board[Side.get_side(side)][hole] += seeds

    def add_to_store(self, side: Side, seeds: int):
        if seeds < 0:
            raise Exception('seeds number is negative')
        self.board[Side.get_side(side)][0] += seeds

    def set_store(self, side: Side, seeds: int):
        if seeds < 0:
            raise Exception('seeds number is negative')
        if side != 1 or side != 0:
            raise Exception('Invalid side')
        self.board[Side.get_side(side)][0] = seeds

    def get_store(self, side: Side):
        if side != 1 or side != 0:
            raise Exception('Invalid side')
        return self.board[Side.get_side(side)][0]

    def get_flipped_board(self):
        copy = Board.clone(self)
        flipped_board = copy.board
        flipped_board[0][0], flipped_board[1][0] = flipped_board[1][0], flipped_board[0][0]
        for hole in range(1, copy.holes + 1):
            flipped_board[0][hole], flipped_board[1][hole] = flipped_board[1][hole], flipped_board[0][hole]
        return flipped_board

    def get_board_image(self, flipped=False):
        if flipped:
            return np.reshape(np.array(self.get_flipped_board()), (2, 8, 1))
        return np.reshape(np.array(self.board), (2, 8, 1))

    def is_seedable(self, side: Side, hole: int) -> bool:
        for other_hole in range(hole - 1, 0):
            if self.get_seeds_in_hole(side, other_hole) == hole - other_hole:
                return True
        return False

    def get_seeds_op(self, side: Side, hole: int):
        if hole < 1 or hole > self.holes:
            raise Exception('incorrect number of holes')
        return self.board[Side.get_index(Side.opposite(side))][self.holes+1-hole]

    def set_seeds_op(self, side: Side, hole: int, seeds: int):
        if hole < 1 or hole > self.holes:
            raise ValueError('Hole number invalid')
        if seeds < 0:
            raise ValueError('Seed number negative')

        self.board[Side.get_index(Side.opposite(side))][self.holes+1-hole] = seeds

    def __str__(self):
        board_str = str(self.board[Side.get_side(Side.NORTH)][0]) + " --"
        for i in range(self.holes, 0, -1):
            board_str += " " + str(self.board[Side.get_side(Side.NORTH)][i])
        board_str += "\n"

        for i in range(1, self.holes + 1, 1):
            board_str += " " + str(self.board[Side.get_side(Side.SOUTH)][i])
        board_str += " --  " + str(self.board[Side.get_side(Side.SOUTH)][0]) + "\n"

        return board_str















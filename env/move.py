from env.side import Side
from copy import deepcopy


class Move(object):
    def __init__(self, side: Side, hole: int):
        if hole < 0 or hole > 7:
            raise Exception("Invalid hole")

        self._side = side
        self._hole = hole

    def clone(self):
        return deepcopy(self)

    @property
    def side(self) -> Side:
        return self._side

    @property
    def hole(self) -> int:
        return self._hole

    def __str__(self) -> str:
        return "Side: %s; Hole: %d" % (Side.to_string(self._side), self._hole)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._side == other._side \
            and Side.get_side(self._side) == Side.get_side(other._side)

    def __hash__(self) -> int:
        return self._hole + (Side.get_side(self._side) * 10)

    def __repr__(self) -> str:
        return "Side: %s; Hole: %d" % (Side.side_to_str(self._side), self._hole)
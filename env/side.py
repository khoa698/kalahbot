from enum import Enum


class Side(Enum):
    NORTH = 0
    SOUTH = 1

    @staticmethod
    def get_side(side) -> int:
        if side is side.NORTH:
            return 0
        return 1

    @classmethod
    def opposite(cls, side):
        if side is side.NORTH:
            return side.SOUTH
        return side.NORTH

    @staticmethod
    def to_string(side) -> str:
        if side is side.NORTH:
            return "North"
        else:
            return "South"
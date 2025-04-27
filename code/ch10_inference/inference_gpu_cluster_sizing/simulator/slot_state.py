from enum import Enum

class SlotState(Enum):
    empty = 0
    prefill = 1
    decoding = 2
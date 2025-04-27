from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Tuple
from .slot_state import SlotState

@dataclass
class Request:
    id: str
    # how long does prefill of this request take, if it is the only request in the engine
    prefill_time: float = 2.
    # how long does decoding of one token of this request take, if it is the only request in the engine
    itl: float = 1.
    # target output length, how many tokens to decode in total to complete the request
    target_output_len_tokens: int = 4

    added_to_queue_at: Optional[float] = None
    started_at: Optional[float] = None
    tokens_generated: int = 0


    def is_in_prefill(self) -> bool:
        return self.tokens_generated == 0

    def get_slot_state_at(self, current_time: float) -> SlotState:
        if self.started_at is None:
            return SlotState.empty
        elif current_time < self.started_at:
            return SlotState.empty
        elif self.is_in_prefill():
            return SlotState.prefill
        elif self.tokens_generated < self.target_output_len_tokens:
            return SlotState.decoding
        else:
            return SlotState.empty

    def get_current_latency_at(self, current_time: float) -> float:
        assert self.added_to_queue_at is not None
        return current_time - self.added_to_queue_at
    
    def tick(self):
        self.tokens_generated += 1

    def get_current_duration(self) -> float:
        if self.is_in_prefill():
            return self.prefill_time
        else:
            return self.itl

@dataclass
class ChunkedContextRequest(Request):
    # into how many chunks in the prefill split
    total_prefill_chunks: int = 4
    prefill_chunks_completed: int = 1

    def tick(self):
        if self.prefill_chunks_completed < self.total_prefill_chunks:
            self.prefill_chunks_completed += 1
        else:
            self.tokens_generated += 1

    def get_current_duration(self) -> float:
        if self.is_in_prefill():
            return self.prefill_time / self.total_prefill_chunks
        else:
            return self.itl
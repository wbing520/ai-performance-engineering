from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Engine

class Metrics:
    engine: 'Engine' 
    num_slots: int
    times: List[float] # time_id -> beginning time of the slot in ticks
    queue_size: List[int]
    e2e_latency: List[Tuple[float, float]] # pairs of time, e2e_latency
    ttft: List[Tuple[float, float]] # pairs of time, ttft
    itl: List[Tuple[float, float]] # pairs of time, itl


    def __init__(self, num_slots, engine):
        self.num_slots = num_slots
        self.engine = engine
        self.times = [0]
        self.queue_size = []
        self.e2e_latency = []
        self.ttft = []
        self.itl = []
        self.osl = [] # target_output_len_tokens

    def track_previous_batch(self):
        for slot, req in self.engine.current_batch.items():
            if req.tokens_generated == 1:
                self.ttft.append((self.engine.current_time, req.get_current_latency_at(self.engine.current_time)))
            if req.tokens_generated == req.target_output_len_tokens:
                self.e2e_latency.append((self.engine.current_time, req.get_current_latency_at(self.engine.current_time)))
                self.osl.append((self.engine.current_time, req.target_output_len_tokens))

    def track_current_batch(self):
        self.queue_size.append(len(self.engine.queue))
        for slot, req in self.engine.current_batch.items():
            if req.tokens_generated > 1:
                self.itl.append((self.engine.current_time, self.engine.get_current_batch_duration()))

    def get_time_interval(self, time_id: int):
        return (self.times[time_id], self.times[time_id+1])

    @classmethod
    def get_values(cls, latencies):
        """Use to get just values of the metrics, without corresponding times. Example: 
        `Metrics.get_values(metrics.itl)`
        """

        return list(latency for time, latency in latencies)

    def get_e2e_latencies(self) -> List[float]:
        return Metrics.get_values(self.e2e_latency)

    def get_ttfts(self) -> List[float]:
        return Metrics.get_values(self.ttft)
    
    def get_itls(self) -> List[float]:
        return Metrics.get_values(self.itl)
    
    def get_osls(self) -> List[float]:
        return Metrics.get_values(self.osl)

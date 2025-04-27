from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Tuple, TYPE_CHECKING
import numpy as np

from .request import Request, ChunkedContextRequest

if TYPE_CHECKING:
    from .engine import Engine

class LoadGenerator:
    engine: 'Engine'
    prefill_time: float = 2
    itl: float = 1
    target_output_len_tokens: int = 4
    total_prefill_chunks: int = 1

    def __init__(self, 
                prefill_time: float = 2, 
                itl: float = 1, 
                target_output_len_tokens: int = 4,
                total_prefill_chunks: int = 1
        ) -> None:
        self.prefill_time = prefill_time
        self.itl = itl
        self.target_output_len_tokens = target_output_len_tokens
        self.total_prefill_chunks = total_prefill_chunks

    def generate_load(self):
        self.engine.queue.append(self.get_request(""))

    def get_request(self, id_postfix):
        return ChunkedContextRequest(
            id=f"{self.engine.current_time}-{id_postfix}",
            prefill_time=self.prefill_time,
            itl=self.itl,
            total_prefill_chunks=self.total_prefill_chunks,
            target_output_len_tokens=self.target_output_len_tokens,
            added_to_queue_at=self.engine.current_time
            )

    def add_n_requests_to_queue(self, n_requests):
        for i in range(n_requests):
            self.engine.queue.append(self.get_request(i))

    def __str__(self):
        res = f"{self.__class__.__name__}("
        for k, v in self.__dict__.items():
            if k not in ['engine', 'last_generation_time']:
                res += f"\n     {k}={v}"
        res += "\n)"
        return res

class LoadGeneratorNormalOutputLength(LoadGenerator):
    target_output_len_std: int = 0

    def get_request(self, id_postfix):
        target_output_len_tokens=max(np.random.normal(self.target_output_len_tokens, self.target_output_len_std), 1)
        return ChunkedContextRequest(
            id=f"{self.engine.current_time}-{id_postfix}",
            prefill_time=self.prefill_time,
            itl=self.itl,
            total_prefill_chunks=self.total_prefill_chunks,
            target_output_len_tokens=int(target_output_len_tokens),
            added_to_queue_at=self.engine.current_time
            )

class BatchLoadGenerator(LoadGeneratorNormalOutputLength):
    initial_batch: int = 30

    def __init__(self, 
                prefill_time: float = 2, 
                itl: float = 1, 
                target_output_len_tokens: int = 4,
                total_prefill_chunks: int = 1,
                initial_batch: int = 3):
        super().__init__(prefill_time, itl, target_output_len_tokens, total_prefill_chunks)
        self.initial_batch = initial_batch

    def generate_load(self):
        if self.engine.current_time > 0:
            return
        self.add_n_requests_to_queue(self.initial_batch)

class ConcurrentLoadGenerator(LoadGeneratorNormalOutputLength):
    target_concurrency: int = 3

    def __init__(self, 
                prefill_time: float = 2, 
                itl: float = 1, 
                target_output_len_tokens: int = 4,
                total_prefill_chunks: int = 1,
                target_concurrency: int = 3):
        super().__init__(prefill_time, itl, target_output_len_tokens, total_prefill_chunks)
        self.target_concurrency = target_concurrency

    def generate_load(self):
        current_concurrency = len(self.engine.get_occupied_slots())
        already_in_queue = len(self.engine.queue)
        ## We want to reach target concurrency but not overshoot it, so limit queue buffer
        need_to_add = self.target_concurrency - current_concurrency - already_in_queue
        self.add_n_requests_to_queue(need_to_add)
        self.last_generation_time = self.engine.current_time

class RequestRateLoadGenerator(LoadGeneratorNormalOutputLength):
    request_rate: float = 1.
    last_generation_time: float = 0

    def __init__(self, 
                 prefill_time: float = 2, 
                 itl: float = 1, 
                 target_output_len_tokens: int = 4, 
                 total_prefill_chunks: int = 1,
                 request_rate: float = 1.
                ):
        super().__init__(prefill_time, itl, target_output_len_tokens, total_prefill_chunks)
        self.request_rate = request_rate

    def generate_load(self):
        generate_every = 1./self.request_rate
        already_generated = self.last_generation_time // generate_every
        final_generated = self.engine.current_time // generate_every
        num_requests = final_generated - already_generated
        self.add_n_requests_to_queue(int(num_requests))
        self.last_generation_time = self.engine.current_time
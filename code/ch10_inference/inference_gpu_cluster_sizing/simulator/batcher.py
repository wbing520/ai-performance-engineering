from typing import List, Optional, Dict, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Engine

class Batcher:
    engine: 'Engine'

    def add_requests(self):
        engine = self.engine
        if engine.get_occupied_slots(): return # static batcher cannot batch together new prefills with old decodings
        for slot in engine.get_all_slots():
            if not len(engine.queue): # checking if we still have more requests to run
                break
            request = engine.queue.pop(0)
            engine.assign_request_to_slot(request, slot)

    def __str__(self):
        return f"{self.__class__.__name__}"

class StaticBatcher(Batcher):
    pass

class IFBatcher(Batcher):
    engine: 'Engine'

    def add_requests(self):
        engine = self.engine
        empty_slots = engine.get_all_slots() - engine.get_occupied_slots()
        for slot in empty_slots:
            if not len(engine.queue):
                break
            request = engine.queue.pop(0)
            engine.assign_request_to_slot(request, slot)

class IFBatcherWithOnePrefillOnly(IFBatcher):
    def add_requests(self):
        engine = self.engine
        # Only one request can be in prefill simultaneously
        if len(engine.get_prefilling_requests()):
            return
        empty_slots = engine.get_all_slots() - engine.get_occupied_slots()
        for slot in empty_slots:
            if not len(engine.queue):
                break
            request = engine.queue.pop(0)
            engine.assign_request_to_slot(request, slot)
            break # Only one new request can be taken
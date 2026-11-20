import time
import tracemalloc


class OverheadProfiler:
    def start_tracking(self):
        # tracemalloc measures Python allocations made during this simulation.
        tracemalloc.start()

    def stop_tracking(self) -> float:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / 1024.0  # KB

    def measure_inference(self, fn, history: list):
        # Wall time is what the simulator charges; CPU time is reported too.
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        result = fn(history)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        cpu_ms = (time.process_time() - cpu_start) * 1000.0
        return result, wall_ms, cpu_ms

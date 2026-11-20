"""A small event-based simulator for the project's CPU scheduling policies."""

from math import isfinite

from src.evaluation.overhead import OverheadProfiler
from src.scheduling.fcfs import select_next_fcfs
from src.scheduling.process import Process
from src.scheduling.round_robin import select_next_rr
from src.scheduling.sjf import select_next_ideal_sjf, select_next_predicted_sjf


class SchedulingSimulator:
    """Run one policy over a list of process dictionaries.

    Each process dictionary has a ``pid``, an ``arrival_time`` in milliseconds,
    and a list of CPU ``bursts``. Between bursts, the simulator models a short
    I/O pause so other processes can run.
    """

    POLICIES = {"fcfs", "rr", "ideal_sjf", "predicted_sjf"}

    def __init__(
        self,
        policy: str,
        predictor=None,
        time_quantum: float = 3.0,
        cs_cost: float = 0.05,
        charge_overhead: bool = True,
        io_delay: float = 1.0,
    ):
        if policy not in self.POLICIES:
            raise ValueError(f"Unknown scheduling policy: {policy}")
        if policy == "predicted_sjf" and predictor is None:
            raise ValueError("predicted_sjf requires a burst predictor")
        if not isfinite(time_quantum) or time_quantum <= 0:
            raise ValueError("time_quantum must be greater than zero")
        if not isfinite(cs_cost) or not isfinite(io_delay) or cs_cost < 0 or io_delay < 0:
            raise ValueError("context-switch cost and I/O delay cannot be negative")

        self.policy = policy
        self.predictor = predictor
        self.time_quantum = time_quantum
        self.cs_cost = cs_cost
        self.charge_overhead = charge_overhead
        self.io_delay = io_delay
        self.profiler = OverheadProfiler()

    def _prepare_burst(self, process: Process):
        """Set the estimate used by SJF and return wall and CPU inference time."""
        if self.policy != "predicted_sjf":
            process.predicted_burst = process.current_true_burst
            return 0.0, 0.0

        estimate, wall_ms, cpu_ms = self.profiler.measure_inference(
            self.predictor.predict_next, process.history
        )
        # Keep the estimate valid even if a different predictor is supplied.
        process.predicted_burst = max(0.1, float(estimate))
        process.predictions.append(process.predicted_burst)
        process.actuals.append(process.current_true_burst)
        if not self.charge_overhead:
            wall_ms = 0.0
        return wall_ms, cpu_ms

    def _select_process(self, ready_queue):
        """Ask the selected policy for the index of the next process."""
        selectors = {
            "fcfs": select_next_fcfs,
            "rr": select_next_rr,
            "ideal_sjf": select_next_ideal_sjf,
            "predicted_sjf": select_next_predicted_sjf,
        }
        return selectors[self.policy](ready_queue)

    @staticmethod
    def _make_processes(raw_workload):
        """Copy and check input records so a run does not change its input."""
        processes = []
        seen_pids = set()
        for item in raw_workload:
            pid = item["pid"]
            arrival = float(item["arrival_time"])
            bursts = [float(burst) for burst in item["bursts"]]
            if pid in seen_pids:
                raise ValueError(f"Process IDs must be unique; found {pid}")
            if not isfinite(arrival) or arrival < 0:
                raise ValueError("arrival_time must be finite and non-negative")
            if not bursts or any(not isfinite(burst) or burst <= 0 for burst in bursts):
                raise ValueError("Each process needs positive CPU burst times")
            seen_pids.add(pid)
            processes.append(Process(pid, arrival, bursts))
        return sorted(processes, key=lambda process: (process.arrival_time, process.pid))

    def run(self, raw_workload: list) -> dict:
        """Simulate the workload and return completed processes and measures."""
        unarrived = self._make_processes(raw_workload)
        ready_queue = []
        # The blocked queue stores (time the process becomes ready, process).
        blocked = []
        completed = []

        sim_time = 0.0
        context_switches = 0
        total_pred_overhead_ms = 0.0
        total_pred_cpu_ms = 0.0
        ordering_inversions = 0
        scheduling_decisions = 0
        last_pid = None

        self.profiler.start_tracking()
        try:
            while unarrived or blocked or ready_queue:
                # A process becomes ready only after both arrival and I/O wait.
                while unarrived and unarrived[0].arrival_time <= sim_time:
                    process = unarrived.pop(0)
                    process.ready_enter_time = process.arrival_time
                    wall_ms, cpu_ms = self._prepare_burst(process)
                    total_pred_overhead_ms += wall_ms
                    total_pred_cpu_ms += cpu_ms
                    sim_time += wall_ms
                    ready_queue.append(process)

                blocked.sort(key=lambda item: (item[0], item[1].pid))
                while blocked and blocked[0][0] <= sim_time:
                    ready_time, process = blocked.pop(0)
                    process.ready_enter_time = ready_time
                    wall_ms, cpu_ms = self._prepare_burst(process)
                    total_pred_overhead_ms += wall_ms
                    total_pred_cpu_ms += cpu_ms
                    sim_time += wall_ms
                    ready_queue.append(process)

                if not ready_queue:
                    # Move time forward to the next arrival or I/O completion.
                    next_events = []
                    if unarrived:
                        next_events.append(unarrived[0].arrival_time)
                    if blocked:
                        next_events.append(blocked[0][0])
                    if next_events:
                        sim_time = max(sim_time, min(next_events))
                    continue

                # Check whether this choice passes up a shorter ready burst.
                shortest_ready = min(process.current_true_burst for process in ready_queue)
                if len(ready_queue) > 1:
                    scheduling_decisions += 1

                selected_index = self._select_process(ready_queue)
                process = ready_queue.pop(selected_index)
                if process.current_true_burst > shortest_ready:
                    ordering_inversions += 1

                # Include the switch delay in the selected process's wait time.
                if last_pid is not None and last_pid != process.pid:
                    context_switches += 1
                    sim_time += self.cs_cost
                last_pid = process.pid
                process.waiting_in_ready_queue += max(
                    0.0, sim_time - process.ready_enter_time
                )

                # Round Robin runs for at most one time quantum per turn.
                run_time = process.remaining_burst
                if self.policy == "rr":
                    run_time = min(self.time_quantum, process.remaining_burst)
                sim_time += run_time
                process.remaining_burst -= run_time

                if process.remaining_burst > 1e-9:
                    # Its quantum expired, so it joins the back of the queue.
                    process.ready_enter_time = sim_time
                    ready_queue.append(process)
                    continue

                # Save metrics for this burst, then give the process its next burst.
                true_burst = process.current_true_burst
                process.total_waiting_time += process.waiting_in_ready_queue
                process.history.append(true_burst)
                process.burst_idx += 1
                process.waiting_in_ready_queue = 0.0

                if process.is_finished:
                    # Report one waiting and turnaround value per process.
                    process.wait_times.append(process.total_waiting_time)
                    process.turnaround_times.append(
                        max(0.0, sim_time - process.arrival_time)
                    )
                    completed.append(process)
                else:
                    process.remaining_burst = process.current_true_burst
                    process.burst_ready_time = sim_time + self.io_delay
                    # Do not predict until the I/O pause has actually finished.
                    blocked.append((process.burst_ready_time, process))
        finally:
            peak_mem_kb = self.profiler.stop_tracking()

        return {
            "completed": completed,
            "context_switches": context_switches,
            "total_pred_overhead_ms": total_pred_overhead_ms,
            "total_pred_cpu_ms": total_pred_cpu_ms,
            "inversion_rate": (
                ordering_inversions / scheduling_decisions
                if scheduling_decisions
                else 0.0
            ),
            "peak_memory_kb": peak_mem_kb,
        }

"""State kept for one process while it moves through the simulator."""

from dataclasses import dataclass, field


@dataclass
class Process:
    pid: int
    arrival_time: float
    bursts: list[float]
    burst_idx: int = 0
    remaining_burst: float = field(init=False)
    predicted_burst: float = 0.0
    ready_enter_time: float = 0.0
    burst_ready_time: float = 0.0
    waiting_in_ready_queue: float = 0.0
    total_waiting_time: float = 0.0
    history: list[float] = field(default_factory=list)
    wait_times: list[float] = field(default_factory=list)
    turnaround_times: list[float] = field(default_factory=list)
    predictions: list[float] = field(default_factory=list)
    actuals: list[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.bursts:
            raise ValueError("A process must have at least one CPU burst")
        self.remaining_burst = self.bursts[0]
        self.ready_enter_time = self.arrival_time
        self.burst_ready_time = self.arrival_time

    @property
    def current_true_burst(self) -> float:
        return self.bursts[self.burst_idx]

    @property
    def is_finished(self) -> bool:
        return self.burst_idx >= len(self.bursts)

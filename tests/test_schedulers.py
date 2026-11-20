import unittest
from src.scheduling.simulator import SchedulingSimulator

class TestSchedulers(unittest.TestCase):
    def setUp(self):
        # Guide example: A=8ms, B=3ms, C=5ms arriving at t=0
        self.sample_workload = [
            {"pid": 0, "arrival_time": 0.0, "bursts": [8.0]},
            {"pid": 1, "arrival_time": 0.0, "bursts": [3.0]},
            {"pid": 2, "arrival_time": 0.0, "bursts": [5.0]},
        ]

    def test_fcfs_waiting_time(self):
        # Waits: A=0, B=8, C=11 -> avg = 19/3 = 6.333 ms
        sim = SchedulingSimulator(policy="fcfs", cs_cost=0.0)
        out = sim.run(self.sample_workload)
        waits = [p.wait_times[0] for p in sorted(out["completed"], key=lambda x: x.pid)]
        self.assertAlmostEqual(sum(waits) / 3.0, 19.0 / 3.0, places=4)

    def test_ideal_sjf_waiting_time(self):
        # Order: B(3), C(5), A(8) -> Waits: B=0, C=3, A=8 -> avg = 11/3 = 3.667 ms
        sim = SchedulingSimulator(policy="ideal_sjf", cs_cost=0.0)
        out = sim.run(self.sample_workload)
        waits = [p.wait_times[0] for p in sorted(out["completed"], key=lambda x: x.pid)]
        self.assertAlmostEqual(sum(waits) / 3.0, 11.0 / 3.0, places=4)

if __name__ == "__main__":
    unittest.main()
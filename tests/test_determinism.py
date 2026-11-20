import unittest

from src.data.generate_workload import generate_workload
from src.scheduling.simulator import SchedulingSimulator


class TestDeterminism(unittest.TestCase):
    def test_identical_seeds_produce_identical_results(self):
        w1 = generate_workload(20, 4, 0.2, seed=7)
        w2 = generate_workload(20, 4, 0.2, seed=7)
        r1 = SchedulingSimulator("ideal_sjf").run(w1)
        r2 = SchedulingSimulator("ideal_sjf").run(w2)
        self.assertEqual(r1["context_switches"], r2["context_switches"])
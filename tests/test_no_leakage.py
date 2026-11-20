import unittest

from src.scheduling.simulator import SchedulingSimulator


class SpyPredictor:
    def __init__(self):
        self.seen_histories = []

    def predict_next(self, history):
        self.seen_histories.append(list(history))
        return 5.0

class TestNoLeakage(unittest.TestCase):
    def test_predictor_only_sees_past_bursts(self):
        workload = [{"pid": 0, "arrival_time": 0.0, "bursts": [11.0, 22.0, 33.0]}]
        spy = SpyPredictor()
        sim = SchedulingSimulator(policy="predicted_sjf", predictor=spy, charge_overhead=False)
        sim.run(workload)
        self.assertEqual(spy.seen_histories, [[], [11.0], [11.0, 22.0]])
"""Selection rules for ideal and predicted Shortest-Job-First scheduling."""


def select_next_ideal_sjf(ready_queue):
    # This is an oracle: it can see the true burst and is an upper benchmark.
    return min(range(len(ready_queue)), key=lambda i: (ready_queue[i].current_true_burst, ready_queue[i].pid))


def select_next_predicted_sjf(ready_queue):
    # The predicted policy sees only a model's estimate, not the true burst.
    return min(range(len(ready_queue)), key=lambda i: (ready_queue[i].predicted_burst, ready_queue[i].pid))

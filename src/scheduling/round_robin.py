"""Selection rule for Round Robin scheduling."""


def select_next_rr(ready_queue):
    # The simulator puts processes that used their time slice at the end.
    return 0

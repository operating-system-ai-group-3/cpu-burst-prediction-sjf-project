"""Selection rule for First-Come, First-Served scheduling."""


def select_next_fcfs(ready_queue):
    # The process that has waited longest goes first; PID breaks ties.
    return min(range(len(ready_queue)), key=lambda i: (ready_queue[i].ready_enter_time, ready_queue[i].pid))

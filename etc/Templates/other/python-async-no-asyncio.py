#!/usr/bin/env python3

"""
    Concurrently execute async functions
    and await statements with no asyncio.
"""

import time
from collections import deque

class kernel:

    def __init__(self, name):
        self.name = name
        self.job = 0

    def __await__(self):
        args = yield self
        self.job += 1
        return f"{self.name}:job={self.job}:args={args}"

scheduler1 = kernel('kernel-a')
scheduler2 = kernel('kernel-b')

async def foo():
    print(f"foo: entering foo")
    val = await scheduler1
    print(f"foo: okay we're back, got {val}")
    val = await scheduler2
    print(f"foo: okay we're back again, got {val}")
    val = await scheduler1
    print(f"foo: okay we're back for the last time, got {val}")
    return 42

async def bar():
    print(f"bar: entering bar")
    val = await scheduler2
    print(f"bar: okay we're back, got {val}")
    val = await scheduler1
    print(f"bar: okay we're back again, got {val}")
    val = await scheduler2
    print(f"bar: okay we're back for the last time, got {val}")
    return 69

coros = [foo(), bar()]
tasks = deque([])
results = []

# initialize tasks to be run
for coro in coros:
    tasks.append(coro)

# run the tasks until they're complete
while True:
    if not tasks:
        break
    coro = tasks.popleft()
    try:
        if coro.cr_await is None:
            awaited = coro.send(None)
        else:
            awaited = coro.send('magic')
        print('awaited:', awaited.name)
    except StopIteration as exc:
        results.append(exc.value)
    else:
        tasks.append(coro)

print(f"results:", results)


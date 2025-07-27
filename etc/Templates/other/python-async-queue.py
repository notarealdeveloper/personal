#!/usr/bin/env python3

import os
import sys
import asyncio


async def say_after(msg, secs):
    await asyncio.sleep(secs)
    print(msg)
    return msg


async def example_tasks():

    print(f"coroutines but not tasks:")
    ret1 = await say_after("Hello", 2)
    ret2 = await say_after("world", 1)
    print(ret1, ret2, end='\n')

    print(f"coroutines and tasks:")
    task1 = asyncio.create_task(say_after("Hello", 2))
    task2 = asyncio.create_task(say_after("world", 1))
    ret1 = await task1
    ret2 = await task2
    print(ret1, ret2, end='\n')


async def worker(n, queue):
    print(f"Entering worker {n}")
    try:
        while True:
            job = await queue.get()
            print(f"Worker {n} got a job!")
            await say_after(*job)
            queue.task_done()
    except asyncio.CancelledError as e:
        print(f"Worker {n} cancelled")
        raise e

async def example_queue():

    queue = asyncio.Queue()

    tasks = [asyncio.create_task(worker(n, queue)) for n in range(5)]

    for (msg, secs) in (('Hello', 1), ('world', 2)):
        await queue.put((msg, secs))

    await queue.join()
    for task in tasks:
        task.cancel()

asyncio.run(example_tasks())
asyncio.run(example_queue())

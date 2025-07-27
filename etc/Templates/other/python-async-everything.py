#!/usr/bin/env python3

import os
import time
import asyncio
import concurrent.futures

processpool = concurrent.futures.ProcessPoolExecutor(os.cpu_count())
threadpool  = concurrent.futures.ThreadPoolExecutor(64)
loop        = asyncio.get_event_loop()
loop.set_default_executor(threadpool)

def bar(n):
    print(f"Running bar({n})")
    time.sleep(1)
    return n**2

async def abar(n):
    return await asyncio.to_thread(bar, n)

async def foo1():
    futures = [loop.run_in_executor(None, bar, n) for n in range(10)]
    return await asyncio.gather(*futures)

async def foo2():
    coros = [asyncio.to_thread(bar, n) for n in range(10)]
    for coro in asyncio.as_completed(coros):
        print(await coro)
    return

async def foo3():
    for fut in asyncio.as_completed(map(abar, range(10))):
        print(await fut)
    return

def run(corofunc, *args, **kwds):
    return loop.run_until_complete(corofunc(*args, **kwds))

run(foo1)
run(foo2)
run(foo3)


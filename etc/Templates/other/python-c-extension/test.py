#!/usr/bin/env python3

import cpython

def foo():
    x = 40
    def bar():
        nonlocal x
        x += 1
        return x
    return bar


bar = foo()

assert bar() == 41
assert bar() == 42
cells = bar.__closure__
assert len(cells) == 1
cell = cells[0]
Cell = type(cell)
cell = Cell(69)
ret = cpython.setclosure(bar, cell)
print(f"ret = {ret}")
assert bar() == 70
assert bar() == 71


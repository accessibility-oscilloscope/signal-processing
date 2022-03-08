#!/usr/bin/env python3
"""
generate a binary representation of a test signal.
"""

import math
import random
import matplotlib.pyplot as plt
import numpy as np

SIGNAL_LENGTH: int = 4000
SIGNAL_AMPLITUDE: int = 255


def f(x: float) -> float:
    return 100 * math.sin(5 * 2 * math.pi * x / SIGNAL_LENGTH) + 127 + random.uniform(-5, 5)


if __name__ == "__main__":
    signal = r"../test.signal"
    fd = open(signal, "wb")

    xs = np.linspace(0, SIGNAL_LENGTH - 1, SIGNAL_LENGTH)
    ys = [f(x) for x in xs]
    ys[240] = 255

    ba = bytearray([int(y) for y in ys])
    print(list(ba))
    fd.write(ba)

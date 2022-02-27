#!/usr/bin/env python3
"""
generate a binary representation of a test signal.
"""

import math, random
import numpy as np

SIGNAL_LENGTH: int = 480
SIGNAL_AMPLITUDE: int = 255


def f(x: float) -> float:
    return 100 * math.sin(5 * 2 * math.pi * x / SIGNAL_LENGTH) + 127 + random.uniform(-5, 5)


if __name__ == "__main__":
    signal = r"C:\Users\Alex\Desktop\Accessible Oscilloscope\Signal Processing\test.signal"
    fd = open(signal, "wb")

    xs = np.linspace(0, SIGNAL_LENGTH - 1, SIGNAL_LENGTH)
    ys = [f(x) for x in xs]
    ys[240] = 255

    fd.write(bytearray([int(y) for y in ys]))

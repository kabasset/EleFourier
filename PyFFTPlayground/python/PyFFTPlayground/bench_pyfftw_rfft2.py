#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare fftconvolve with scipy and pyfftw backends
   Adapted from https://github.com/pyFFTW/pyFFTW/issues/310#issuecomment-946342809
"""

import pyfftw
import multiprocessing
import scipy.signal
from scipy.fft import rfft2, set_backend
import numpy as np
from timeit import Timer


def main():
    xsize = 1024
    ysize = 1024
    number_it = 100

    a = pyfftw.empty_aligned((xsize, ysize), dtype=np.float64)
    b = pyfftw.empty_aligned((xsize, ysize), dtype=np.float64)

    a[:] = np.random.randn(xsize, ysize)
    b[:] = np.random.randn(xsize, ysize)

    t = Timer(lambda: run_rfft2(a, b))

    print(f"Time with scipy.fft default backend installed: {(t.timeit(number=number_it)):1.3f} seconds")

    # Configure PyFFTW to use all cores (the default is single-threaded)
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"

    # Use the backend pyfftw.interfaces.scipy_fft
    with set_backend(pyfftw.interfaces.scipy_fft, only=True):
        # Turn on the cache for optimum performance
        pyfftw.interfaces.cache.enable()

        # We cheat a bit by doing the planning first
        run_rfft2(a, b)

        print(f"Time with pyfftw backend installed: {(t.timeit(number=number_it)):1.3f} seconds")


def run_rfft2(a, b):
    b = rfft2(a)
    return b


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""A simple python script to test pyfftw with parallel computing in Python.
"""


import sys
from typing import Tuple, Any
import pyfftw
from functools import partial
import numpy as np

# from joblib import Parallel, delayed
import multiprocessing as mp

from Timer import Timer

# class DFT(pyfftw.FFTW):
#     def __getnewargs__(self):
#         print("__getnewargs__")
#         return (self.input_array, self.output_array, self.axes, self.direction, self.flags)

#     def __getstate__(self):
#         return ()

#     def __setstate__(self, state):
#         pass

#     def __getnewargs_ex__(self):
#         print("__getnewargs_ex__")
#         return (self.input_array, self.output_array, self.axes, self.direction, self.flags)


def create_complex_plan(shape: Tuple[int], flags: Tuple[str]) -> "pyfftw.FFTW":
    """Create plan for monochromatic PSF computation. It is a complex inverse DFT
        Parameters
        ----------
        shape : tuple of ints
           shape of the input and output DFT

        flags : tuple of strings
           pyfftw planning flags
    """
    i = pyfftw.empty_aligned(shape, dtype="complex128")
    o = pyfftw.empty_aligned(shape, dtype="complex128")
    return pyfftw.FFTW(i, o, flags=flags)


def apply_dft(loop: int, wisdom: Tuple[str], param: int) -> "pyfftw.FFTW":
    # print(f"Generate DFT for parameter {param}")
    pyfftw.import_wisdom(wisdom)

    input = gen_rand_comp_array(1024, 1024)
    for i in range(loop):
        input *= i
        chrono = Timer()
        chrono.start()
        dft = pyfftw.FFTW(input, pyfftw.empty_aligned((1024, 1024), dtype="complex128"), flags=("FFTW_WISDOM_ONLY",))
        output = dft()
        chrono.stop()
        # print(f"# Elapsed time for DFT lambda {i} of parameter {param}: {chrono.elapsed_time:.3f} milliseconds")


def gen_rand_comp_array(sizex: int, sizey: int) -> "np.ndarray":
    """Generate a random array of complex value (to simulate input pupil for instance)

        Parameters
        ----------
        sizex : int
           image side
        sizey : int
           image side

        Returns
        -------
        np.ndarray:
          ndarray of random complex numbers (real part generated from the "standard normal” distribution)
        """

    return np.random.randn(sizex, sizey) + np.random.randn(sizex, sizey) * 1j


def main():
    # Generate pyfftw plan
    params = 40
    input_shp = (1024, 1024)
    flags = ("FFTW_MEASURE",)

    main_chrono = Timer()
    main_chrono.start()
    planning = create_complex_plan(input_shp, flags)
    main_chrono.stop()
    print(f"# Elapsed time for planning: {main_chrono.elapsed_time:.3f} milliseconds")
    wisdom = pyfftw.export_wisdom()

    # plans = [create_complex_plan(input_shp, flags) for p in range(params)]

    param = [p for p in range(params)]

    with mp.Pool(3) as p:
        p.map(partial(apply_dft, 10, wisdom), param)

    # pool = Pool(nodes=1)
    # pool.map(partial(apply_dft, 10), plans)

    # Parallel(n_jobs=2, prefer="processes")(delayed(partial(apply_dft, 10))(p) for p in plans)


if __name__ == "__main__":
    mp.set_start_method("fork")
    sys.exit(main())

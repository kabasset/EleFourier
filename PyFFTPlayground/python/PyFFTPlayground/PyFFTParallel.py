#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#


"""
:file: python/PyFFTPlayground/PyFFTParallel.py

:date: 02/22/22
:author: user

"""

import argparse
from socket import MSG_PEEK
from typing import Tuple, Any
import numpy as np
from functools import partial

# Use "concurrent.futures" instead of multiprocessing as pyfftw Cython classes do no implement __reduce__ (pickled)
# See https://stackoverflow.com/questions/69032946/python-process-typeerror-no-default-reduce-due-to-non-trivial-cinit
# Can't use ProcessPool also with "concurrent.futures" as it relies also on pickled object
# There is an issue in pyfftw tracker which ask to pyfftw to add support for pickle in FFTW class
# See https://github.com/pyFFTW/pyFFTW/issues/130
# Using ThreadPoolExecutor
import multiprocessing as mp
import pyfftw

import ElementsKernel.Logging as log
import ElementsKernel.Exit as Exit

from PyFFTPlayground.DFT import create_complex_plan, create_real_plan, create_real_plan_backward
from PyFFTPlayground.Timer import Timer


class BranchDFTs:
    """The set of DFT plans of each parallel branch.

    Attributes
    ----------
    - pupil_to_psf : PyFFTPlayground.DFT
    - psf_to_mtf : PyFFTPlayground.DFT
    - mtf_to_broadband : PyFFTPlayground.DFT
    - lambdas: int
        Number of monochromatic PSF to generate
    - chrono: Timer
        Chronometer to monitor DFTs in each branches
    """

    def __init__(self, mask_side: int, psf_side: int, flags: Tuple[str]):
        self.pupil_to_psf = create_complex_plan((mask_side, mask_side), flags)
        self.psf_to_mtf = create_real_plan((mask_side, mask_side), flags)
        self.mtf_to_broadband = create_real_plan_backward((psf_side, psf_side), flags)
        self.chrono = Timer()


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


def dummy_psf_broadband(
    pupil: "np.ndarray", mask_side: int, psf_side: int, lambdas: int, wisdom: Tuple[str], flags: Tuple[str], param: int
) -> Tuple[Any, Timer]:
    """Function to simulate pd computation. Loop over number of lambda value and compute 3 DFTs sequentially, sum all
    mono PSF and apply a single inverse DFT to compute broadband PSF

        Parameters
        ----------
        pupil : np.ndarray
           Complex pupil function
        plan : BranchDFTs
           Set of FFTW plans

        Returns
        -------
        tuple:
          PSF broadband image and the chronometer associated to the computation
        """

    # TODO It does not seem necessary to import wisdom as it is shared between process context TBC
    # TODO Remove wisdom argument
    # pyfftw.import_wisdom(wisdom)
    wisdom_flag = flags + ("FFTW_WISDOM_ONLY",)
    # Shortcuts for BranchDFTs attributes
    plan = BranchDFTs(mask_side=mask_side, psf_side=psf_side, flags=wisdom_flag)

    dft_pupil_to_psf = plan.pupil_to_psf
    psf_to_mtf = plan.psf_to_mtf
    mtf_to_broadband = plan.mtf_to_broadband
    chrono = plan.chrono
    chrono.start()
    # Store sum of mono DFT (use to simulate computation of broadband PSF)
    mtf_shape = psf_to_mtf.output_array.shape
    mtf_dtype = psf_to_mtf.output_array.dtype
    mtf_broad_sum = np.zeros(mtf_shape, dtype=mtf_dtype)

    # DFT for each lambda
    for l in range(lambdas):
        # Compute DFT
        # chrono.start()
        # TODO Check if __call__ should be used here to update input array of FFTW plan
        # See https://pyfftw.readthedocs.io/en/latest/source/pyfftw/pyfftw.html#pyfftw.FFTW.__call__
        psf_mono = dft_pupil_to_psf(input_array=pupil)
        # chrono.stop()

        # chrono.start()
        # Multiply by random array (should be pupil * exponential(-2jPi/WFE)...)
        wfe = gen_rand_comp_array(psf_mono.shape[0], psf_mono.shape[1]) * l
        psf_mono *= wfe
        # Compute norm of mono psf (to apply DFT on real numbers)
        abs_psf_mono = np.abs(psf_mono)
        # chrono.stop()

        # chrono.start()
        mtf = psf_to_mtf(input_array=abs_psf_mono)
        # chrono.stop()

        # TODO Implement a more realistic way to merge each mono DFT
        # Update sum of mtf
        mtf_broad_sum += mtf

    # chrono.start()
    # Convert lambdas arrays of size mask_side*mask_side to one array of size broadband_side*broadband_side
    # TODO Implement a more realistic generation of mono to multi broadband
    mtf_broad_sum /= lambdas

    # Slice mtf_broad_sum array to ensure that its shape is compatible with the mtf_to_broadband FFTW plan
    # Dummy selection here to simulate broadband of the correct shape
    broadband_shape = mtf_to_broadband.input_array.shape
    mtf_broad_sum = mtf_broad_sum[: broadband_shape[0], : broadband_shape[1]]
    # chrono.stop()

    # Then compute the MTF broadband (inverse DFT)
    # chrono.start()
    broadband = mtf_to_broadband(input_array=mtf_broad_sum)

    chrono.stop()

    return broadband, chrono


def defineSpecificProgramOptions():
    """
    @brief Allows to define the (command line and configuration file) options
    specific to this program

    @details See the Elements documentation for more details.
    @return An  ArgumentParser.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-f", "--flags", nargs="+", default=["FFTW_MEASURE"], help="List of strings for flags used by FFTW planners"
    )
    parser.add_argument("-p", "--params", type=int, default=1, help="Number of parameters")
    parser.add_argument("-b", "--branches", type=int, default=1, help="Number of process in parallel")
    parser.add_argument("-l", "--lambdas", type=int, default=1, help="Number of wavelengths")
    parser.add_argument("-m", "--maskside", type=int, default=1024, help="Side size of input mask")
    parser.add_argument("-psf", "--psfside", type=int, default=512, help="Output PSF side size")
    parser.add_argument("--print", action="store_true", help="Print detailed information of DFTs computation time.")

    return parser


def mainMethod(args):
    """
    @brief The "main" method.

    @details This method is the entry point to the program. In this sense, it is
    similar to a main (and it is why it is called mainMethod()).
    """

    logger = log.getLogger("PyFFTParallel")

    logger.info("#")
    logger.info("# Entering PyFFTParallel mainMethod()")
    logger.info("#")

    # Read parameters
    flags = tuple(args.flags)
    branches = args.branches
    params = args.params
    lambdas = args.lambdas
    mask_side = args.maskside
    broadband_side = args.psfside

    main_chrono = Timer()

    logger.info("# Initialize FFTW plans sequentially...")
    main_chrono.start()
    # Planning
    plans = BranchDFTs(mask_side, broadband_side, flags)
    wisdom = pyfftw.export_wisdom()
    main_chrono.stop()
    logger.info(f"# Elapsed time for planning: {main_chrono.elapsed_time:.3f} milliseconds")

    main_chrono.start()
    # Generate random data for input pupil mask
    pupil = gen_rand_comp_array(mask_side, mask_side)
    param_list = [x for x in range(params)]

    # Use plan in parallel with multiprocessing
    with mp.Pool(processes=branches) as pool:
        # Use functools partial function for the fixed input pupil mask
        # Suggested here https://stackoverflow.com/a/49358837
        results = pool.map(
            partial(dummy_psf_broadband, pupil, mask_side, broadband_side, lambdas, wisdom, flags), param_list
        )

        if args.print:
            for i, broadband in enumerate(results):
                chrono = broadband[1]
                logger.info("#")
                logger.info(f"# Profiling of DFTs for parameter {i}:")
                logger.info("# Timing per DFT:")
                logger.info([f"{incs:.3f} milliseconds" for incs in chrono.incs])
                logger.info(
                    f"# Total elapsed time for PSF broadband execution {i}: {chrono.elapsed_time:.3f} milliseconds"
                )
                logger.info("#")

    main_chrono.stop()

    logger.info("#")
    logger.info(
        f"# Total elapsed time for planning + DFTs in parallel branches: {main_chrono.elapsed_time:.3f} milliseconds"
    )
    logger.info("#")

    return Exit.Code["OK"]

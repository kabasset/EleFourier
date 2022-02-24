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
import ElementsKernel.Logging as log
import ElementsKernel.Exit as Exit
import numpy as np

from PyFFTPlayground.DFT import create_complex_plan, create_real_plan, create_real_plan_backward
from PyFFTPlayground.Timer import Timer


class BranchDFTs:
    def __init__(self, mask_side, psf_side, flags, logger):
        self.pupil_to_psf = create_complex_plan((mask_side, mask_side), flags)
        self.psf_to_mtf = create_real_plan((mask_side, mask_side), flags)
        self.mtf_to_broadband = create_real_plan_backward((psf_side, psf_side), flags)
        self.chrono = Timer(logger=logger.info)


def defineSpecificProgramOptions():
    """
    @brief Allows to define the (command line and configuration file) options
    specific to this program

    @details See the Elements documentation for more details.
    @return An  ArgumentParser.
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--flags", nargs="+", default=["FFTW_MEASURE"], help="List of strings for flags used by FFTW planners"
    )
    parser.add_argument("--params", type=int, default=1, help="Number of parameters")
    parser.add_argument("--lambdas", type=int, default=1, help="Number of wavelengths")
    parser.add_argument("--maskside", type=int, default=1024, help="Side size of input mask")
    parser.add_argument("--psfside", type=int, default=512, help="Output PSF side size")

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
    params = args.params
    lambdas = args.lambdas
    mask_side = args.maskside
    broadband_side = args.psfside
    main_chrono = Timer(logger=logger.info)

    side_ratio = mask_side // broadband_side

    # FIXME Implement for params > 1
    if params > 1:
        logger.info("# Only params == 1 is supported")
        logger.info("# Exiting PyFFTParallel mainMethod()")
        return Exit.Code["USAGE"]

    logger.info("# Initialize FFTW plans sequentially...")
    main_chrono.start()
    plans = [BranchDFTs(mask_side, broadband_side, flags, logger) for p in range(params)]
    main_chrono.stop()

    for plan in plans:
        # Shortcuts
        dft_pupil_to_psf = plan.pupil_to_psf.plan
        psf_to_mtf = plan.psf_to_mtf.plan
        mtf_to_broadband = plan.mtf_to_broadband.plan
        chrono = plan.chrono

        # Store sum of mono DFT (use to simulate computation of broadband PSF)
        mtf_broad_sum = np.zeros(psf_to_mtf.output_array.shape, dtype=psf_to_mtf.output_array.dtype)
        # DFT for each lambda
        for l in range(lambdas):
            # Generate random data as input
            input = np.random.randn(dft_pupil_to_psf.input_array.shape[0], dft_pupil_to_psf.input_array.shape[1])
            # Compute DFT
            logger.info("# Compute Pupil to PSF DFT...")
            chrono.start()
            # FIXME Not sure if __call__ should be used here to update input array of FFTW plan
            # See https://pyfftw.readthedocs.io/en/latest/source/pyfftw/pyfftw.html#pyfftw.FFTW.__call__
            psf_mono = dft_pupil_to_psf(input_array=input)
            chrono.stop()

            # Compute norm of mono psf (to apply real DFT)
            abs_psf_mono = np.abs(psf_mono)

            logger.info("# Compute Pupil to MTF DFT...")
            chrono.start()
            mtf = psf_to_mtf(input_array=abs_psf_mono)
            chrono.stop()

            # TODO Better way to merge each mono DFT
            # Update sum of mtf
            mtf_broad_sum += mtf

        # Convert lambdas arrays of size mask_side*mask_side to one array of size broadband_side*broadband_side
        # FIXME Wrong method probably here as mtf_broad_sum is of size mask.shape//2 + 1
        mtf_broad_sum /= lambdas
        broadband_shape = mtf_to_broadband.input_array.shape
        mtf_broad_sum = mtf_broad_sum[: broadband_shape[0], : broadband_shape[1]]

        # Then compute the MTF broadband (inverse)
        logger.info("# Compute DFT broadband...")
        chrono.start()
        broadband = mtf_to_broadband(input_array=mtf_broad_sum)
        chrono.stop()

    logger.info("#")
    logger.info("# Exiting PyFFTParallel mainMethod()")
    logger.info("#")

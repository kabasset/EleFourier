/**
 * @copyright (C) 2012-2020 Euclid Science Ground Segment
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 3.0 of the License, or (at your option)
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 *
 */

#include "EleFits/SifFile.h"
#include "EleFourier/Zernike.h"

#include <boost/test/unit_test.hpp>

using namespace Euclid;

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE(Zernike_test)

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(save_zernikes_test) {
  constexpr long diameter = 1024;
  Fits::VecRaster<double, 3> zernike({diameter, diameter, 20});
  for (long y = 0; y < zernike.length<1>(); ++y) {
    for (long x = 0; x < zernike.length<0>(); ++x) {
      Fourier::LocalZernikeSeries series(x, y, 0.5 * diameter, 0);
      const auto values = series.ansiSeq<20>();
      for (long z = 0; z < zernike.length<2>(); ++z) {
        zernike[{x, y, z}] = values[z];
      }
    }
  }
  Fits::SifFile f("/tmp/zernike.fits", Fits::FileMode::Overwrite);
  f.writeRaster(zernike);
}

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE_END()

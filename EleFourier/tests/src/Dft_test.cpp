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

#include "EleFits/MefFile.h" // FIXME rm
#include "EleFourier/Dft.h"

#include <boost/test/unit_test.hpp>

using namespace Euclid;
using namespace Fourier;

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE(Dft_test)

//-----------------------------------------------------------------------------

template <typename T>
void checkFftwMalloc(long width, long height, long count) {
  auto raster = Internal::makeFftwRaster<T>(width, height, count);
  BOOST_TEST(raster.size() == width * height * count);
  BOOST_TEST(raster.data() != nullptr);
  auto begin = raster.data();
  auto end = begin + raster.size();
  std::fill(begin, end, T(1));
  for (const auto& p : raster.domain()) {
    BOOST_TEST(raster[p] == T(1));
  }
  fftw_free(raster.data());
}

BOOST_AUTO_TEST_CASE(fftw_malloc_test) {
  constexpr long width = 3;
  constexpr long height = 14;
  constexpr long count = 2;
  checkFftwMalloc<double>(width, height, count);
  checkFftwMalloc<std::complex<double>>(width / 2 + 1, height, count);
}

BOOST_AUTO_TEST_CASE(fftw_real_malloc_test) {
  constexpr long width = 5;
  constexpr long height = 6;
  constexpr long count = 3;
  auto image = Internal::makeFftwRaster<double>(width, height, count);
  for (const auto& p : image.domain()) {
    image[p] = 1 + p[0] + p[1] + p[2];
  }
  auto coefs = Internal::makeFftwRaster<std::complex<double>>(width / 2 + 1, height, count);
  auto transform = Internal::makeFftwTransform<RealToComplex>(image, coefs);
  auto inverse = Internal::makeFftwInverse<RealToComplex>(image, coefs);
  fftw_execute(transform); // Coefficients computed, image is junk
  fftw_execute(inverse); // Image recomputed, coefficients are junk
  fftw_destroy_plan(transform);
  fftw_destroy_plan(inverse);
  for (const auto& p : image.domain()) {
    const auto expected = width * height * (1 + p[0] + p[1] + p[2]);
    const auto value = image[p];
    BOOST_TEST(value > 0.99 * expected);
    BOOST_TEST(value < 1.01 * expected);
  }
  fftw_free(image.data());
  fftw_free(coefs.data());
}

BOOST_AUTO_TEST_CASE(example_test) {
  Fits::Position<2> shape {30, 14};
  RealDft dft(shape);
  BOOST_TEST(dft.image().shape() == shape);
  for (const auto& p : dft.image().domain()) {
    dft.image()[p] = 1 + p[0] + 2 * p[1];
  }
  Fits::MefFile f("/tmp/dft.fits", Fits::FileMode::Overwrite);
  printf("Writing image...\n");
  f.assignImageExt("IN", dft.image());
  printf("Transforming...\n");
  dft.transform();
  printf("Filling conjugates...\n");
  const auto full = dft.evalConjugates();
  printf("Writing magnitude...\n");
  f.assignImageExt("OUT", evalMagnitude(full));
  printf("Inverting...\n");
  dft.inverse();
  printf("Writing image...\n");
  for (const auto& p : dft.image().domain()) {
    const auto expected = dft.normalizationFactor() * (1 + p[0] + 2 * p[1]);
    const auto value = dft.image()[p];
    BOOST_TEST(value > 0.99 * expected);
    BOOST_TEST(value < 1.01 * expected);
  }
  f.assignImageExt("IN2", dft.image());
}

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE_END()

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

#include "EleFourier/Dft.h"

#include <boost/test/unit_test.hpp>

using namespace Euclid;
using namespace Fourier;

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE(Dft_test)

//-----------------------------------------------------------------------------

/*
template <typename T>
void checkFftwMalloc(long width, long height, long count) {
  auto raster = Internal::makeFftwRaster<T>(width, height, count);
  BOOST_TEST(raster.size() == width * height * count);
  BOOST_TEST(raster.data() != nullptr);
  std::fill(raster.begin(), raster.end(), T(1));
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

BOOST_AUTO_TEST_CASE(fftw_r2c_c2r_test) {
  constexpr long width = 5;
  constexpr long height = 6;
  constexpr long count = 3;
  auto signal = Internal::makeFftwRaster<double>(width, height, count);
  auto fourier = Internal::makeFftwRaster<std::complex<double>>(width / 2 + 1, height, count);
  auto transform = Internal::makeFftwTransform<RealToComplex>(signal, fourier);
  auto inverse = Internal::makeFftwInverse<RealToComplex>(signal, fourier);
  for (const auto& p : signal.domain()) {
    signal[p] = 1 + p[0] + p[1] + p[2];
  }
  fftw_execute(transform); // Coefficients computed, signal is junk
  fftw_execute(inverse); // Image recomputed, coefficients are junk
  fftw_destroy_plan(transform);
  fftw_destroy_plan(inverse);
  for (const auto& p : signal.domain()) {
    const auto expected = width * height * (1 + p[0] + p[1] + p[2]);
    const auto value = signal[p];
    BOOST_TEST(value > 0.99 * expected);
    BOOST_TEST(value < 1.01 * expected);
  }
  fftw_free(signal.data());
  fftw_free(fourier.data());
}
*/

BOOST_AUTO_TEST_CASE(dft_r2c_c2r_test) {

  // Initialize plans
  constexpr long width = 5;
  constexpr long height = 6;
  constexpr long count = 3;
  RealForwardDft r2c({width, height}, count);
  auto c2r = r2c.inverse();

  // Fill signal
  for (long i = 0; i < count; ++i) {
    auto signal = r2c.inBuffer(i);
    for (const auto& p : signal.domain()) {
      signal[p] = 1 + p[0] + p[1] + i;
    }
  }

  // Apply and then inverse
  r2c.transform();
  c2r.transform().normalize();

  // Check values are recovered
  for (long i = 0; i < count; ++i) {
    auto signal0 = r2c.inBuffer(i);
    auto signal1 = c2r.outBuffer(i);
    for (const auto& p : signal0.domain()) {
      const auto expected = 1L + p[0] + p[1] + i;
      const auto value = signal0[p];
      BOOST_TEST(value > 0.99 * expected);
      BOOST_TEST(value < 1.01 * expected);
      BOOST_TEST(signal1[p] == value);
    }
  }
}

/*
BOOST_AUTO_TEST_CASE(dft_c2c_test) {
  // Initialize plans
  constexpr long width = 5;
  constexpr long height = 6;
  constexpr long count = 3;
  ComplexDft c2c({width, height}, count);

  // Fill signal
  for (long i = 0; i < count; ++i) {
    auto signal = c2c.signal(i);
    for (const auto& p : signal.domain()) {
      signal[p] = {1. + p[0], 1. + p[1] + i};
    }
  }

  // Apply and then inverse
  c2c.apply();
  c2c.inverseNormalize();

  // Check values are recovered
  for (long i = 0; i < count; ++i) {
    auto signal = c2c.signal(i);
    for (const auto& p : signal.domain()) {
      const std::complex<double> expected {1. + p[0], 1. + p[1] + i};
      const auto value = signal[p];
      BOOST_TEST(value.real() > 0.99 * expected.real());
      BOOST_TEST(value.real() < 1.01 * expected.real());
      BOOST_TEST(value.imag() > 0.99 * expected.imag());
      BOOST_TEST(value.imag() < 1.01 * expected.imag());
    }
  }
}

BOOST_AUTO_TEST_CASE(dft_r2c_c2c_c2r_test) {
  // Initialize plans
  constexpr long width = 5;
  constexpr long height = 6;
  constexpr long count = 3;
  RealDft r2c({width, height}, count);
  auto c2c = r2c.compose();
  
  // Fill r2c signal
  for (long i = 0; i < count; ++i) {
    auto signal = r2c.signal(i);
    for (const auto& p : signal.domain()) {
      signal[p] = 1 + p[0] + p[1] + i;
    }
  }

  // Apply r2c and then c2c
  printf("Fill r2c.fourier() aka c2c.signal()\n");
  r2c.apply();
  printf("Fill c2c.fourier()\n");
  c2c.apply();

  // Inverse c2c and then r2c
  printf("Fill c2c.signal() aka r2c.fourier()\n");
  c2c.inverseNormalize();
  printf("Fill r2c.signal()\n");
  r2c.inverseNormalize();

  // Check values are recovered
  for (long i = 0; i < count; ++i) {
    auto signal = r2c.signal(i);
    for (const auto& p : signal.domain()) {
      const auto expected = 1 + p[0] + p[1] + i; // No scaling anymore
      const auto value = signal[p];
      BOOST_TEST(value > 0.99 * expected);
      BOOST_TEST(value < 1.01 * expected);
    }
  }
}
*/

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE_END()

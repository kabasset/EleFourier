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
#include <type_traits>

using namespace Euclid;
using namespace Fourier;

//-----------------------------------------------------------------------------

BOOST_AUTO_TEST_SUITE(Dft_test)

//-----------------------------------------------------------------------------

template <typename T>
void checkFftwMalloc(const Fits::Position<2>& shape, long count) {
  auto raster = RealForwardDft::initFftwBuffer<T, false>(shape, count, nullptr);
  BOOST_TEST(raster.size() == shapeSize(shape) * count);
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
  checkFftwMalloc<double>({width, height}, count);
  checkFftwMalloc<std::complex<double>>({width / 2 + 1, height}, count);
}

BOOST_AUTO_TEST_CASE(fftw_r2c_malloc_test) {
  Fits::Position<2> shape {3, 4};
  long count = 5;
  auto in = RealForwardDft::initFftwBuffer<double, false>(shape, count, nullptr);
  auto out = RealForwardDft::initFftwBuffer<std::complex<double>, false>(shape, count, nullptr);
  auto plan = RealForwardDftType::initFftwPlan(in, out);
  fftw_free(in.data());
  fftw_free(out.data());
  fftw_destroy_plan(plan);
  fftw_cleanup();
}

/*
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

template <typename TTransform>
void checkInit(const Fits::Position<2>& shape, long count) {
  TransformPlan<typename TTransform::Type, false, false> plan(shape, count);
  BOOST_TEST(plan.shape() == shape);
  BOOST_TEST(plan.count() == count);
  BOOST_TEST(plan.inShape() == TTransform::Type::Inverse::outShape(shape));
  BOOST_TEST(plan.outShape() == TTransform::Type::outShape(shape));
  BOOST_TEST(plan.inBuffer().data() != nullptr);
  BOOST_TEST(plan.outBuffer().data() != nullptr);
}

BOOST_AUTO_TEST_CASE(init_test) {
  const Fits::Position<2> shape {4, 3};
  const long count = 10;
  checkInit<RealForwardDft>(shape, count);
  checkInit<RealBackwardDft>(shape, count);
  checkInit<ComplexForwardDft>(shape, count);
  checkInit<ComplexBackwardDft>(shape, count);
}

template <typename TTransform>
void checkInverse(const Fits::Position<2>& shape, long count) {
  TransformPlan<typename TTransform::Type, false, false> plan(shape, count);
  auto inverse = plan.inverse();
  using Expected = typename decltype(plan)::Type::Inverse;
  using Inverse = typename decltype(inverse)::Type;
  BOOST_TEST((std::is_same<Inverse, Expected>::value));
  BOOST_TEST(inverse.shape() == shape);
  BOOST_TEST(inverse.count() == count);
  BOOST_TEST(inverse.inShape() == plan.outShape());
  BOOST_TEST(inverse.outShape() == plan.inShape());
  BOOST_TEST(inverse.inBuffer().data() == plan.outBuffer().data());
  BOOST_TEST(inverse.outBuffer().data() == plan.inBuffer().data());
}

BOOST_AUTO_TEST_CASE(inverse_test) {
  const Fits::Position<2> shape {4, 3};
  const long count = 10;
  checkInverse<RealForwardDft>(shape, count);
  checkInverse<RealBackwardDft>(shape, count);
  checkInverse<ComplexForwardDft>(shape, count);
  checkInverse<ComplexBackwardDft>(shape, count);
}

template <typename TTransform, typename UTransform>
void checkComposition() {
  const Fits::Position<2> shape {4, 3};
  const long count = 10;
  TransformPlan<typename TTransform::Type, false, false> plan(shape, count);
  auto composed = plan.template compose<UTransform>();
  BOOST_TEST(composed.shape() == shape);
  BOOST_TEST(composed.count() == count);
  BOOST_TEST(composed.inShape() == plan.outShape());
  BOOST_TEST(composed.inBuffer().data() == plan.outBuffer().data());
  BOOST_TEST(composed.outBuffer().data() != nullptr);
}

BOOST_AUTO_TEST_CASE(real_composition_test) {
  checkComposition<RealForwardDft, RealBackwardDft>();
  checkComposition<RealBackwardDft, RealForwardDft>();
}

BOOST_AUTO_TEST_CASE(real_complex_composition_test) {
  checkComposition<RealForwardDft, ComplexForwardDft>();
  checkComposition<ComplexBackwardDft, RealBackwardDft>();
}

BOOST_AUTO_TEST_CASE(complex_composition_test) {
  checkComposition<ComplexForwardDft, ComplexBackwardDft>();
  checkComposition<ComplexBackwardDft, ComplexForwardDft>();
}

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
    auto signal = r2c.inBuffer(i);
    BOOST_TEST(c2r.outBuffer(i).data() == signal.data());
    for (const auto& p : signal.domain()) {
      const auto expected = (1L + p[0] + p[1] + i) * r2c.normalizationFactor();
      const auto value = signal[p];
      BOOST_TEST(value > 0.99 * expected);
      BOOST_TEST(value < 1.01 * expected);
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

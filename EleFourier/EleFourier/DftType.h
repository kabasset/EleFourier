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

#ifndef _ELEFOURIER_DFTTYPE_H
#define _ELEFOURIER_DFTTYPE_H

#include "EleFitsData/Raster.h"

#include <complex>
#include <fftw3.h>

namespace Euclid {
namespace Fourier {

template <typename TType>
struct Inverse;

/**
 * @brief Base DFT type to be inherited.
 */
template <typename TType, typename TIn, typename TOut>
struct DftType {

  /**
   * @brief The parent `DftType` class.
   */
  using Parent = DftType;

  /**
   * @brief The tag.
   */
  using Type = TType;

  /**
   * @brief The input value type.
   */
  using InValue = TIn;

  /**
   * @brief The output value type.
   */
  using OutValue = TOut;

  /**
   * @brief The tag of the inverse transform.
   */
  using InverseType = Inverse<TType>;

  /**
   * @brief Input buffer shape.
   * @param shape The logical shape
   */
  static Fits::Position<2> inShape(const Fits::Position<2>& shape) {
    return shape;
  }

  /**
   * @brief Output buffer shape.
   * @param shape The logical shape
   */
  static Fits::Position<2> outShape(const Fits::Position<2>& shape) {
    return shape;
  }
};

template <typename TType, typename TIn, typename TOut>
struct DftType<Inverse<TType>, TIn, TOut> {

  using Parent = DftType;
  using Type = Inverse<TType>;
  using InValue = TOut;
  using OutValue = TIn;
  using InverseType = TType;

  static Fits::Position<2> inShape(const Fits::Position<2>& shape) {
    return DftType<TType, TIn, TOut>::outShape(shape);
  }

  static Fits::Position<2> outShape(const Fits::Position<2>& shape) {
    return DftType<TType, TIn, TOut>::inShape(shape);
  }
};

template <typename TType>
struct Inverse : DftType<Inverse<TType>, typename TType::InValue, typename TType::OutValue> {};

/**
 * @brief Real DFT type.
 */
struct RealDftType;
struct RealDftType : DftType<RealDftType, double, std::complex<double>> {};
template <>
Fits::Position<2> RealDftType::Parent::outShape(const Fits::Position<2>& shape);

/**
 * @brief Complex DFT type.
 */
struct ComplexDftType;
struct ComplexDftType : DftType<ComplexDftType, std::complex<double>, std::complex<double>> {};

/**
 * @brief Complex DFT type with Hermitian symmertry.
 */
struct HermitianComplexDftType;
struct HermitianComplexDftType : DftType<HermitianComplexDftType, std::complex<double>, std::complex<double>> {};
template <>
Fits::Position<2> HermitianComplexDftType::Parent::inShape(const Fits::Position<2>& shape);
template <>
Fits::Position<2> HermitianComplexDftType::Parent::outShape(const Fits::Position<2>& shape);

template <typename TType>
fftw_plan initFftwPlan(
    Fits::PtrRaster<typename TType::InValue, 3>& in,
    typename Fits::PtrRaster<typename TType::OutValue, 3>& out);

} // namespace Fourier
} // namespace Euclid

#endif

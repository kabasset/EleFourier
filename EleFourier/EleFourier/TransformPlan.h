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

#ifndef _ELEFOURIER_TRANSFORMPLAN_H
#define _ELEFOURIER_TRANSFORMPLAN_H

#include "EleFitsData/Raster.h"

#include <cassert>
#include <fftw3.h>

namespace Euclid {
namespace Fourier {

/**
 * @brief Singleton class to be instantiated by FFTW user class constructors
 * to ensure proper cleanup at program ending.
 * @details
 * The destructor, which is executed once, at the end of the program, calls `fftw_cleanup()`.
 */
class FftwGlobalsCleaner {
private:
  /**
   * @brief Private constructor.
   */
  FftwGlobalsCleaner() {}

public:
  /**
   * @brief Destructor.
   * @details
   * Frees FFTW's globals.
   */
  ~FftwGlobalsCleaner() {
    fftw_cleanup();
  }

  /**
   * @brief Instantiate the singleton, to trigger cleanup at destruction.
   */
  static FftwGlobalsCleaner& instance() {
    static FftwGlobalsCleaner instance;
    return instance;
  }
};

/**
 * @brief Memory- and computation-efficient discrete Fourier transform.
 * 
 * @details
 * This class provides a light wrapping of FFTW's transforms.
 * It is design to compose transforms (e.g. direct and inverse DFTs) efficiently.
 * 
 * On memory side, one plan comes with an input buffer and an output buffer,
 * which are allocated at plan construction.
 * Obviously, it is optimal to work directly in the buffers,
 * and avoid performing copies before and after transforms.
 * 
 * The most classical use case is to call the forward transform and later the inverse transform.
 * To this end, `inverse()` creates an inverse plan with shared buffers (see example below).
 * It is also possible to pipe transforms with `compose()`.
 * 
 * Another classical use case is to perform the same transform on several inputs.
 * For this purpose, 3D stacks of 2D data are stored in the buffers instead of a single 2D data,
 * which is controlled by the `count` parameter in constructor.
 * 
 * On computation side, the class relies on user-triggered evaluation instead of early or lazy evaluations,
 * i.e. the user has to explicitely call `transform()` when relevant.
 * The sequence is as follows:
 * - At construction, buffers are used to optimize the plan;
 * - The user fills the input buffer;
 * - The user calls `transform()` - now, the input buffer is garbage
 * - The user reads the output buffer.
 * 
 * Here is a classical example to perform a convolution in Fourier domain:
 * 
 * \code
 * RealForwardDft dft(shape);
 * auto inverseDft = dft.inverse();
 * dft.inBuffer() = ... ; // Assign data somehow
 * dft.transform(); // Perform direct transform - dft.inBuffer() = inverseDft.outBuffer() is garbage now
 * const auto& coefficients = dft.outBuffer();
 * ... // Use coefficients, e.g. to convolve by a filter kernel
 * inverseDft.transform().normalize(); // Perform inverse transform - dft.outBuffer() = inverseDft.inBuffer() is garbage now
 * const auto& filtered = inverseDft.outBuffer();
 * ... // Do something with filtered, which contains the convolved signal
 * \endcode
 * 
 * Computation follows FFTW's conventions on formats and scaling, i.e.:
 * - If a buffer has Hermitian symmetry, it is of size `(width / 2 + 1) * height` instead of `width * height`;
 * - None of the transforms are scaled, which means that a factor `width * height` is introduced
 *   by calling `transform()` and then `inverse().transform()` - `normalize()` performs normalization on request.
 */
template <typename TType, bool ShareIn, bool ShareOut>
class TransformPlan {
  template <typename, bool, bool>
  friend class TransformPlan;

public:
  /**
   * @brief The plan type.
   */
  using Type = TType;

  /**
   * @brief The signal value type.
   */
  using InValue = typename TType::InValue;

  /**
   * @brief The Fourier coefficient type.
   */
  using OutValue = typename TType::OutValue;

private:
  /**
   * @brief Constructor.
   * @param shape The logical plane shape
   * @param count The number of planes
   * @param inData The pre-existing input buffer, or `nullptr` to allocate a new one
   * @param outData The pre-existing output buffer, or `nullptr` to allocate a new one
   */
  TransformPlan(Fits::Position<2> shape, long count, InValue* inData, OutValue* outData) :
      m_shape {shape}, m_inShape {TType::Inverse::outShape(shape)}, m_outShape {TType::outShape(shape)},
      m_count {count}, m_in {initFftwBuffer<InValue, ShareIn>(m_inShape, m_count, inData)},
      m_out {initFftwBuffer<OutValue, ShareOut>(m_outShape, m_count, outData)},
      m_plan {TType::initFftwPlan(m_in, m_out)} {
    assert(ShareIn == bool(inData));
    assert(ShareOut == bool(outData));
  }

public:
  template <typename T, bool share>
  static Fits::PtrRaster<T, 3>
  initFftwBuffer(const Fits::Position<2>& shape, long count, T* data) { // FIXME move outside?
    T* d = data ? data : (T*)fftw_malloc(sizeof(T) * shapeSize(shape) * count);
    if (data) {
      printf("Data already allocated at %p\n", (void*)d);
    } else {
      printf("Allocated %li values at %p\n", shapeSize(shape) * count, (void*)d);
    }
    return {{shape[0], shape[1], count}, d};
  }

  static void freeGlobals() {
    fftw_cleanup();
  }

public:
  /**
   * @brief Constructor.
   * @param shape The logical plane shape
   * @param count The number of planes
   */
  TransformPlan(Fits::Position<2> shape, long count = 1) : TransformPlan(shape, count, nullptr, nullptr) {
    assert(not ShareIn);
    assert(not ShareOut);
  }

  /**
   * @brief Create the inverse `TransformPlan` with shared buffers.
   * @details
   * \code
   * auto planB = planA.inverse();
   * planA.transform(); // Fills planA.outBuffer() = planB.inBuffer()
   * planB.transform().normalize(); // Fills planB.outBuffer() = planA.inBuffer()
   * \endcode
   * @warning
   * This plan (`planA` from the snippet) is the owner of the buffers, which will be freed by its destructor,
   * which means that the buffers of the inverse plan (`planB`) has the same life cycle.
   */
  TransformPlan<typename Type::Inverse, true, true> inverse() {
    return {m_shape, m_count, m_out.data(), m_in.data()};
  }

  /**
   * @brief Create a `TransformPlan` which shares its input buffer with this `TransformPlan`'s output buffer.
   * @details
   * \code
   * auto planB = planA.compose<ComplexForwardDft>();
   * planA.transform(); // Fills planA.outBuffer() = planB.inBuffer()
   * planB.transform();
   * \endcode
   * @warning
   * This plan (`planA` from the snippet) is the owner of its output buffer, which will be freed by its destructor,
   * which means that the input buffer of the composed plan (`planB`) has the same life cycle.
   */
  template <typename TPlan>
  TransformPlan<typename TPlan::Type, true, false> compose() {
    return {
        m_shape,
        m_count,
        m_out.data(),
        nullptr}; // FIXME not necessarily m_shape, e.g. RealForwardDft -> ComplexForwardDft
  }

  /**
   * @brief Destructor.
   * @warning
   * Buffers are freed.
   * If data has to outlive the `TransformPlan` object, buffers should be copied beforehand.
   */
  ~TransformPlan() {
    fftw_destroy_plan(m_plan);
    if (not ShareIn) {
      printf("Freeing input at: %p\n", (void*)m_in.data());
      fftw_free(m_in.data());
    }
    if (not ShareOut) {
      printf("Freeing output: %p\n", (void*)m_out.data());
      fftw_free(m_out.data());
    }
    FftwGlobalsCleaner::instance();
  }

  /**
   * @brief Get the number of planes.
   */
  long count() const {
    return m_count;
  }

  /**
   * @brief Get the logical plane shape.
   */
  const Fits::Position<2>& shape() const {
    return m_shape;
  }

  /**
   * @brief Get the input buffer shape.
   */
  const Fits::Position<2>& inShape() const {
    return m_inShape;
  }

  /**
   * @brief Access the input buffer.
   * @warning
   * Contains garbage after `execute()` has been called.
   */
  const Fits::PtrRaster<const InValue> inBuffer(long index = 0) const {
    return m_in.section(index);
  }

  /**
   * @copydoc inBuffer()
   */
  Fits::PtrRaster<InValue> inBuffer(long index = 0) {
    return m_in.section(index);
  }

  /**
   * @brief Get the output buffer shape.
   */
  const Fits::Position<2>& outShape() const {
    return m_outShape;
  }

  /**
   * @brief Access the output buffer.
   */
  const Fits::PtrRaster<const OutValue> outBuffer(long index = 0) const {
    return m_out.section(index);
  }

  /**
   * @copydoc outBuffer()
   */
  Fits::PtrRaster<OutValue> outBuffer(long index = 0) {
    return m_out.section(index);
  }

  /**
   * @brief Get the normalization factor.
   */
  double normalizationFactor() const {
    return m_shape[0] * m_shape[1];
  }

  /**
   * @brief Compute the transform.
   */
  TransformPlan& transform() {
    fftw_execute(m_plan);
    return *this;
  }

  /**
   * @brief Divide by the output buffer by the normalization factor.
   */
  TransformPlan& normalize() {
    const auto factor = normalizationFactor();
    for (auto& c : m_in) {
      c /= factor;
    }
    return *this;
  }

private:
  /**
   * @brief The logical shape.
   */
  Fits::Position<2> m_shape;

  /**
   * @brief The input shape.
   */
  Fits::Position<2> m_inShape;

  /**
   * @brief The output shape.
   */
  Fits::Position<2> m_outShape;

  /**
   * @brief The number of planes.
   */
  long m_count;

  /**
   * @brief The input stack.
   */
  Fits::PtrRaster<InValue, 3> m_in;

  /**
   * @brief The output stack.
   */
  Fits::PtrRaster<OutValue, 3> m_out;

  /**
   * @brief The transform plan.
   */
  fftw_plan m_plan;
};

} // namespace Fourier
} // namespace Euclid

#endif

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

#ifndef _ELEFOURIER_DFT_H
#define _ELEFOURIER_DFT_H

#include "EleFitsData/Raster.h"

#include <complex>
#include <fftw3.h>

namespace Euclid {
namespace Fourier {

struct ComplexToComplex {
  using Value = std::complex<double>;
  using Coef = std::complex<double>;
};

struct RealToComplex {
  using Value = double;
  using Coef = std::complex<double>;
};

namespace Internal {

template <typename T>
Fits::PtrRaster<T, 3> makeFftwRaster(long width, long height, long count) {
  T* data = (T*)fftw_malloc(sizeof(T) * width * height * count);
  return {{width, height, count}, data};
}

template <typename TDomains>
fftw_plan
makeFftwTransform(Fits::Raster<typename TDomains::Value, 3>& image, Fits::Raster<typename TDomains::Coef, 3>& coefs);

template <typename TDomains>
fftw_plan
makeFftwInverse(Fits::Raster<typename TDomains::Value, 3>& image, Fits::Raster<typename TDomains::Coef, 3>& coefs);

template <typename TDomains>
long coefsWidth(long imageWidth);

} // namespace Internal

/**
 * @brief Memory- and computation-efficient discrete Fourier transform.
 * 
 * @details
 * This class provides a light wrapping of FFTW's `r2c` and `c2r` transforms.
 * It is optimized to work with a single `RealDft` for both transforming the image,
 * and later inverse the coefficients back, i.e.:
 * 
 * \code
 * RealDft dft(shape);
 * dft.image() = ... ; // Assign data somehow
 * dft.transform();
 * const auto& coefs = dft.coefficients();
 * ... // Use coefs, e.g. to convolve by a filter
 * dft.inverse();
 * ... // Do something with dft.image(), which contains the convolved image
 * 
 * On computation side, it relies on user-triggered evaluation instead of early or lazy evaluations,
 * i.e. the user has to explicitely call `transform()` or `inverse()`
 * to trigger the evaluation of the Fourier transform or inverse Fourier transform, respectively.
 * FFTW's forward and backward plans are created at construction.
 * 
 * Computation follows FFTW's conventions on formats and scaling, i.e.:
 * - If the image raster is of size `width * height`, the coefficients raster is of size `(width / 2 + 1) * height`;
 * - None of the transforms are not scaled, which means that
 *   a factor `width * height` is introduced by calling `transform()` and then `inverse()`.
 * 
 * On memory side, rasters are immediately allocated to hold both the image and Fourier coefficients.
 * Calling `transform()` writes in the coefficients raster,
 * while calling `inverse()` writes in the image raster.
 * This allows toggling between the the space and frequency domains without copies.
 * 
 * In addition, several images of identical shapes can be transformed at once,
 * by instantiating a single DFT through the `count` parameter.
 */
template <typename TDomains>
class Dft {

public:
  /**
   * @brief The image value type.
   */
  using Value = typename TDomains::Value;

  /**
   * @brief The Fourier coefficient type.
   */
  using Coef = typename TDomains::Coef;

  /**
   * @brief Constructor.
   * @param shape The shape of each input image
   * @param count The number of input images
   */
  Dft(Fits::Position<2> shape, long count = 1) :
      m_width {shape[0]}, m_coefsWidth {Internal::coefsWidth<TDomains>(m_width)}, m_height {shape[1]}, m_count {count},
      m_image {Internal::makeFftwRaster<Value>(m_width, m_height, m_count)},
      m_coefs {Internal::makeFftwRaster<Coef>(m_coefsWidth, m_height, m_count)},
      m_transform {Internal::makeFftwTransform<TDomains>(m_image, m_coefs)},
      m_inverse {Internal::makeFftwInverse<TDomains>(m_image, m_coefs)} {}

  /**
   * @brief Destructor.
   */
  ~Dft() {
    fftw_free(m_image.data());
    fftw_free(m_coefs.data());
    fftw_destroy_plan(m_transform);
    fftw_destroy_plan(m_inverse);
  }

  /**
   * @brief Access the space domain image.
   */
  const Fits::PtrRaster<const Value> image(long index = 0) const {
    return m_image.section(index);
  }

  /**
   * @copydoc image()
   */
  Fits::PtrRaster<Value> image(long index = 0) {
    return m_image.section(index);
  }

  /**
   * @brief Access the Fourier coefficients.
   */
  const Fits::PtrRaster<const Coef> coefs(long index = 0) const {
    return m_coefs.section(index);
  }

  /**
   * @copydoc coefs()
   */
  Fits::PtrRaster<Coef> coefs(long index = 0) {
    return m_coefs.section(index);
  }

  /**
   * @brief Get the normalization factor.
   */
  double normalizationFactor() const {
    return m_width * m_height;
  }

  /**
   * @brief Compute the Fourier transform.
   * @details
   * The space domain image must have been assigned before calling this function.
   * @warning
   * The image data may be modified, too, for optimizing internal computations.
   */
  void transform() {
    fftw_execute(m_transform);
  }

  /**
   * @brief Compute the inverse Fourier transform.
   * @details
   * The frequency domain image must have been assigned before calling this function.
   * @warning
   * The Fourier coefficients may be modified, too, for optimizing internal computations.
   */
  void inverse() {
    fftw_execute(m_inverse);
  }

  /**
   * @brief Compute the full coefficient raster,
   * including conjugate coefficients not computed by `transform()`.
   * @note
   * This method is only provided for visualization and saving purposes,
   * as the computed conjugates are redundant and therefore not needed for further computations.
   * Expect no optimization efforts in its implementation.
   */
  Fits::VecRaster<Coef> evalConjugates(long index = 0) { // FIXME not for ComplexToComplex
    Fits::VecRaster<Coef> res({m_width, m_height});
    const auto dft = coefs(index);
    for (const auto& p : dft.domain()) {
      res[p] = dft[p];
    }
    Fits::Region<2> right {{m_coefsWidth, 0}, {m_width - 1, m_height - 1}};
    Fits::Position<2> q;
    for (const auto& p : right) {
      q[0] = m_width - p[0] - 1;
      q[1] = m_height - p[1] - 1;
      res[p] = std::conj(dft[q]);
    }
    return res;
  }

  /**
   * @brief Compute the full coefficient raster where zero-frequency is in the center.
   * @details
   * Coefficients are shifted by `width / 2` and `height / 2`, respectively.
   */
  Fits::VecRaster<Coef> centerEvalConjugates(long index = 0) { // FIXME not for ComplexToComplex
    const auto xShift = m_width - m_coefsWidth;
    const auto yShift = m_height - (m_height / 2 + 1); // FIXME check and simplify
    Fits::VecRaster<Coef> res({m_width, m_height});
    const auto coefs = coefs(index);
    Fits::Position<2> q;
    for (const auto& p : coefs.domain()) {
      q = {(p[0] + xShift) % m_width, (p[1] + yShift) % m_height};
      res[q] = coefs[p];
      q = {(m_width - q[0]) % m_width, (m_height - q[1]) % m_height};
      res[q] = std::conj(coefs[p]);
    }
    return res;
  }

private:
  /**
   * @brief The image width.
   */
  long m_width;

  /**
   * @brief The coefficients raster width.
   */
  long m_coefsWidth;

  /**
   * @brief The image and coefficients raster width.
   */
  long m_height;

  /**
   * @brief The number of images.
   */
  long m_count;

  /**
   * @brief The image stack.
   */
  Fits::PtrRaster<Value, 3> m_image;

  /**
   * @brief The coefficients raster stack.
   */
  Fits::PtrRaster<Coef, 3> m_coefs;

  /**
   * @brief The direct transform plan.
   */
  fftw_plan m_transform;

  /**
   * @brief The inverse transform plan.
   */
  fftw_plan m_inverse;
};

/**
 * @brief Compute the Fourier coefficients magnitude.
 */
Fits::VecRaster<double> evalMagnitude(const Fits::Raster<std::complex<double>>& coefs) {
  Fits::VecRaster<double> res(coefs.shape());
  std::transform(coefs.begin(), coefs.end(), res.begin(), [](auto c) {
    return std::norm(c);
  });
  return res;
}

using RealDft = Dft<RealToComplex>;
using ComplexDft = Dft<ComplexToComplex>;

} // namespace Fourier
} // namespace Euclid

#endif

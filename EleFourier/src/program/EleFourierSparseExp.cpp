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
#include "EleFitsData/Raster.h"
#include "EleFitsData/TestRaster.h"
#include "EleFitsUtils/ProgramOptions.h"
#include "EleFitsValidation/Chronometer.h"
#include "EleFourier/Dft.h"
#include "ElementsKernel/ProgramHeaders.h"

#include <complex>
#include <map>
#include <string>

using boost::program_options::value;

using namespace Euclid;

/**
 * @brief Generate Zernike polynomials for each point and each index.
 * @details
 * The axes are ordered as (lambda, u, v) for performance:
 * for each point, indices are contiguous in memory.
 */
Fits::VecRaster<double, 3> generateZernike(long maskSide, long count) {
  const Fits::Position<3> shape = {count, maskSide, maskSide};
  Fits::Test::RandomRaster<double, 3> zernike(shape, 0, 1);
  return zernike;
}

/**
 * @brief Generate a circular pupil mask.
 */
Fits::VecRaster<double> generatePupil(long maskSide, long pupilRadius) {
  Fits::VecRaster<double> pupil({maskSide, maskSide});
  const auto maskRadius = maskSide / 2;
  const auto pupilRadiusSquared = pupilRadius * pupilRadius;
  for (const auto& p : pupil.domain()) {
    const auto u = double(p[0] - maskRadius);
    const auto v = double(p[1] - maskRadius);
    if (std::abs(u * u + v * v) < pupilRadiusSquared) {
      pupil[p] = 1.;
    }
  }
  return pupil;
}

template <typename TRaster>
void saveSif(const TRaster& raster, const std::string& filename) {
  if (filename == "") {
    return;
  }
  Fits::SifFile f(filename, Fits::FileMode::Overwrite);
  f.writeRaster(raster);
}

/**
 * @brief Monochromatic data buffers.
 */
struct MonochromaticData {

  double minusTwoPiOverLambda;
  std::vector<double> alphas;
  Fourier::ComplexDft pupilToPsf;
  Fits::PtrRaster<std::complex<double>> pupil;
  Fits::PtrRaster<std::complex<double>> amplitude;
  Fits::VecRaster<double> intensity;

  MonochromaticData(double lambda, long maskSide, std::vector<double> alphaGuesses) :
      minusTwoPiOverLambda(-2 * 3.1415926 / lambda), alphas(std::move(alphaGuesses)), pupilToPsf({maskSide, maskSide}),
      pupil(pupilToPsf.inBuffer()), amplitude(pupilToPsf.outBuffer()), intensity({maskSide, maskSide}) {}

  std::complex<double> computeLocalPhase(double mask, const double* zernikes) {
    double sum = 0;
    auto zIt = zernikes;
    for (auto aIt = alphas.begin(); aIt != alphas.end(); ++aIt, ++zIt) {
      sum += (*aIt) * (*zIt);
    }
    return mask * std::exp(std::complex<double>(0, minusTwoPiOverLambda * sum));
  }

  template <typename TP, typename TZ>
  Fits::PtrRaster<std::complex<double>>& evalCompletePupil(const TP& mask, const TZ& zernikes) {
    auto maskData = mask.data();
    auto zernikesData = zernikes.data();
    const auto size = alphas.size();
    for (auto it = pupil.begin(); it != pupil.end(); ++it, ++maskData, zernikesData += size) {
      *it = computeLocalPhase(*maskData, zernikesData);
    }
    return pupil;
  }

  template <typename TP, typename TZ>
  Fits::PtrRaster<std::complex<double>>& evalSparsePupil(const TP& mask, const TZ& zernikes) {
    auto maskData = mask.data();
    auto zernikesData = zernikes.data();
    const auto size = alphas.size();
    for (auto it = pupil.begin(); it != pupil.end(); ++it, ++maskData, zernikesData += size) {
      if (*maskData != 0) {
        *it = computeLocalPhase(*maskData, zernikesData);
      } else {
        *it = 0;
      }
    }
    return pupil;
  }

  Fits::PtrRaster<std::complex<double>>& evalAmplitude() {
    pupilToPsf.transform();
    return amplitude;
  }

  Fits::VecRaster<double>& evalIntensity() {
    intensity.generate(
        [](const auto& amp) {
          return std::norm(amp);
        },
        amplitude);
    return intensity;
  }
};

class EleFourierSparseExp : public Elements::Program {

public:
  std::pair<OptionsDescription, PositionalOptionsDescription> defineProgramArguments() override {
    Fits::ProgramOptions options("Compare complete and sparse exponentiations.");
    options.named("side", value<long>()->default_value(1024), "Pupil mask side");
    // FIXME options.named("side", 1024L, "Pupil radius");
    options.named("radius", value<long>()->default_value(512), "Pupil radius");
    options.named("alphas", value<long>()->default_value(40), "Number of Zernike indices");

    options.named("mask", value<std::string>()->default_value("/tmp/mask.fits"), "Pupil mask file");
    options.named("zernike", value<std::string>()->default_value("/tmp/zernike.fits"), "Zernike polynomials file");
    options.named("psf", value<std::string>()->default_value("/tmp/psf.fits"), "PSF file");

    options.flag("sparse", "Compute pupil only where mask is not null");
    return options.asPair();
  }

  ExitCode mainMethod(std::map<std::string, VariableValue>& args) override {
    Logging logger = Logging::getLogger("EleFourierSparseExp");

    const auto maskSide = args["side"].as<long>();
    const auto pupilRadius = args["radius"].as<long>();
    const auto alphaCount = args["alphas"].as<long>();
    const auto maskFilename = args["mask"].as<std::string>();
    const auto zernikeFilename = args["zernike"].as<std::string>();
    const auto psfFilename = args["psf"].as<std::string>();
    const auto sparse = args["sparse"].as<bool>();

    using Chrono = Fits::Validation::Chronometer<std::chrono::milliseconds>;
    Chrono chrono;

    logger.info("Generating pupil mask...");
    // FIXME load if exists, generate and save otherwise
    chrono.start();
    auto pupil = generatePupil(maskSide, pupilRadius);
    chrono.stop();
    logger.info() << "  " << chrono.last().count() << "ms";
    saveSif(pupil, maskFilename);

    logger.info("Generating Zernike polynomials...");
    chrono.start();
    auto zernike = generateZernike(maskSide, alphaCount);
    chrono.stop();
    logger.info() << "  " << chrono.last().count() << "ms";
    saveSif(zernike, zernikeFilename);

    logger.info("Generating Zernike coefficients...");
    chrono.start();
    std::vector<double> alphas(alphaCount, 1);
    chrono.stop();
    logger.info() << "  " << chrono.last().count() << "ms";

    logger.info("Planning DFT and allocating memory...");
    chrono.start();
    MonochromaticData data(500., maskSide, alphas);
    chrono.stop();
    logger.info() << "  " << chrono.last().count() << "ms";

    chrono.start();
    if (sparse) {
      logger.info("Computing pupil amplitude over non zero points (complex exp)...");
      data.evalSparsePupil(pupil, zernike);
    } else {
      logger.info("Computing pupil amplitude over all points (complex exp)...");
      data.evalCompletePupil(pupil, zernike);
    }
    chrono.stop();
    logger.info() << "  " << chrono.last().count() << "ms";

    logger.info("Computing PSF amplitude (complex DFT)...");
    chrono.start();
    data.evalAmplitude();
    chrono.stop();
    logger.info() << "  " << chrono.last().count() << "ms";

    logger.info("Computing PSF intensity (norm)...");
    chrono.start();
    chrono.stop();
    logger.info() << "  " << chrono.last().count() << "ms";
    saveSif(data.evalIntensity(), psfFilename);

    logger.info("Done.");
    return ExitCode::OK;
  }
};

MAIN_FOR(EleFourierSparseExp)

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

/**
 * @brief Monochromatic data buffers.
 */
struct MonochromaticData {

  double minusTwoPiOverLambda;
  std::vector<double> alphas;
  Fits::VecRaster<std::complex<double>> amplitude;
  Fits::VecRaster<double> intensity;

  MonochromaticData(double lambda, long maskSide, std::vector<double> alphaGuesses) :
      minusTwoPiOverLambda(-2 * 3.1415926 / lambda), alphas(std::move(alphaGuesses)), amplitude({maskSide, maskSide}),
      intensity({maskSide, maskSide}) {}

  std::complex<double> computeLocalPhase(double pupil, const double* zernikes) {
    double sum = 0;
    auto zIt = zernikes;
    for (auto aIt = alphas.begin(); aIt != alphas.end(); ++aIt, ++zIt) {
      sum += (*aIt) * (*zIt);
    }
    return pupil * std::exp(std::complex<double>(0, minusTwoPiOverLambda * sum));
  }

  template <typename TP, typename TZ>
  Fits::VecRaster<std::complex<double>>& evalCompleteAmplitude(const TP& pupil, const TZ& zernikes) {
    auto pupilData = pupil.data();
    auto zernikesData = zernikes.data();
    const auto size = alphas.size();
    for (auto ampIt = amplitude.begin(); ampIt != amplitude.end(); ++ampIt, ++pupilData, zernikesData += size) {
      *ampIt = computeLocalPhase(*pupilData, zernikesData);
    }
    return amplitude;
  }

  template <typename TP, typename TZ>
  Fits::VecRaster<std::complex<double>>& evalSparseAmplitude(const TP& pupil, const TZ& zernikes) {
    auto pupilData = pupil.data();
    auto zernikesData = zernikes.data();
    const auto size = alphas.size();
    for (auto ampIt = amplitude.begin(); ampIt != amplitude.end(); ++ampIt, ++pupilData, zernikesData += size) {
      if (*pupilData != 0) {
        *ampIt = computeLocalPhase(*pupilData, zernikesData);
      } else {
        *ampIt = 0;
      }
    }
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

template <typename TRaster>
void saveSif(const TRaster& raster, const std::string& filename) {
  if (filename == "") {
    return;
  }
  Fits::SifFile f(filename, Fits::FileMode::Overwrite);
  f.writeRaster(raster);
}

class EleFourierSparseExp : public Elements::Program {

public:
  std::pair<OptionsDescription, PositionalOptionsDescription> defineProgramArguments() override {
    Fits::ProgramOptions options("Compare complete and sparse exponentiations.");
    options.named("side", value<long>()->default_value(1024), "Pupil mask side");
    options.named("radius", value<long>()->default_value(512), "Pupil radius");
    // FIXME options.named("radius", 1024L, "Pupil radius");
    options.named("alphas", value<long>()->default_value(40), "Number of Zernike indices");
    options.named("pupil", value<std::string>()->default_value("/tmp/pupil.fits"), "Pupil mask input or output file");
    // FIXME options.named("pupil", "/tmp/pupil.fits"s, "Pupil mask input or output file");
    options.named(
        "zernike",
        value<std::string>()->default_value("/tmp/zernike.fits"),
        "Zernike polynomials input or output file");
    options.named("psf", value<std::string>()->default_value("/tmp/psf.fits"), "PSF input or output file");
    options.flag("sparse", "Compute exp only where mask is not null");
    return options.asPair();
  }

  ExitCode mainMethod(std::map<std::string, VariableValue>& args) override {
    Logging logger = Logging::getLogger("EleFourierSparseExp");

    const auto maskSide = args["side"].as<long>();
    const auto pupilRadius = args["radius"].as<long>();
    const auto alphaCount = args["alphas"].as<long>();
    const auto pupilFilename = args["pupil"].as<std::string>();
    const auto zernikeFilename = args["zernike"].as<std::string>();
    const auto psfFilename = args["psf"].as<std::string>();
    const auto sparse = args["sparse"].as<bool>();

    logger.info("Generating pupil mask...");
    // FIXME load if exists, generate and save otherwise
    auto pupil = generatePupil(maskSide, pupilRadius);
    saveSif(pupil, pupilFilename);

    logger.info("Generating Zernike polynomials...");
    auto zernike = generateZernike(maskSide, alphaCount);
    saveSif(zernike, zernikeFilename);

    logger.info("Generating Zernike coefficients...");
    std::vector<double> alphas(alphaCount, 1);

    logger.info("Allocating memory...");
    MonochromaticData data(500., maskSide, alphas);
    if (sparse) {
      logger.info("Computing PSF amplitude over non zero points...");
      data.evalSparseAmplitude(pupil, zernike);
    } else {
      logger.info("Computing PSF amplitude over all points...");
      data.evalCompleteAmplitude(pupil, zernike);
    }
    logger.info("Computing PSF intensity over all points...");
    saveSif(data.evalIntensity(), psfFilename);

    logger.info("Done.");
    return ExitCode::OK;
  }
};

MAIN_FOR(EleFourierSparseExp)

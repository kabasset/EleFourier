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

#include "EleFitsUtils/ProgramOptions.h"
#include "EleFitsValidation/Chronometer.h"
#include "EleFourier/Dft.h"
#include "ElementsKernel/ProgramHeaders.h"

#include <chrono>
#include <functional>
#include <map>
#include <omp.h>
#include <random>
#include <string>

using boost::program_options::value;

using namespace Euclid;
using namespace Fourier;

/**
 * @brief The set of DFT plans of each parallel branch.
 * @warning
 * Plans should be initialized in a single thread, as this writes in global variables.
 */
struct BranchDfts {

  RealDft pupilToPsf;
  RealDft psfToMtf;
  ComplexDft::Inverse psfsToBroadband;

  /** Branch-wise chronometer. */
  Fits::Validation::Chronometer<std::chrono::milliseconds> chrono;

  /** Constructor. */
  BranchDfts(const Fits::Position<2>& pupilShape, const Fits::Position<2>& broadbandShape) :
      pupilToPsf(pupilShape), psfToMtf(pupilShape), psfsToBroadband(broadbandShape), chrono() {}
};

class EleFourierParallelizationTutorial : public Elements::Program {

public:
  std::pair<OptionsDescription, PositionalOptionsDescription> defineProgramArguments() override {
    Fits::ProgramOptions options("Demonstrates multi-thread usage of the library.");
    options.named("params", value<long>()->default_value(40), "Number of parameters");
    options.named("branches", value<long>()->default_value(1), "Number of branches");
    options.named("lambdas", value<long>()->default_value(40), "Number of wavelengths per branch");
    options.named("pupil", value<long>()->default_value(1024), "Input pupil mask side");
    options.named("psf", value<long>()->default_value(512), "Output PSF side (oversampled)");
    return options.asPair();
  }

  Elements::ExitCode mainMethod(std::map<std::string, VariableValue>& args) override {

    Elements::Logging logger = Elements::Logging::getLogger("EleFourierParallelizationTutorial");

    const auto params = args["params"].as<long>();
    const auto branches = args["branches"].as<long>();
    const auto lambdas = args["lambdas"].as<long>();
    const auto pupilSide = args["pupil"].as<long>();
    const auto broadbandSide = args["psf"].as<long>();
    const long sideRatio = pupilSide / broadbandSide;
    const Fits::Position<2> pupilShape {pupilSide, pupilSide};
    const Fits::Position<2> broadbandShape {broadbandSide, broadbandSide};
    Fits::Validation::Chronometer<std::chrono::milliseconds> programChrono;

    // Initialize DFT plans sequentially - NOT THREAD SAFE!
    std::vector<BranchDfts> dfts;
    dfts.reserve(params);
    for (long i = 0; i < params; ++i) {
      logger.info() << "Initializing param #" << i << "'s plans...";
      programChrono.start();
      dfts.emplace_back(pupilShape, broadbandShape);
      programChrono.stop();
      logger.info() << "  Done in " << programChrono.last().count() << " ms.";
    }

    // Use plans in parallel
    logger.info() << "Executing plans in parallel...";
    logger.info() << "  Number of parameters: " << params;
    logger.info() << "  Number of branches: " << branches;
    logger.info() << "  Available number of threads: " << omp_get_max_threads();
    omp_set_num_threads(branches);
    programChrono.start();
#pragma omp parallel for
    for (long i = 0; i < params; ++i) {

      // Shortcuts
      auto& pupilToPsf = dfts[i].pupilToPsf;
      auto& psfToMtf = dfts[i].psfToMtf;
      auto& psfsToBroadband = dfts[i].psfsToBroadband;
      auto& chrono = dfts[i].chrono;
      auto pupilBegin = pupilToPsf.inBuffer().begin();
      auto pupilEnd = pupilToPsf.inBuffer().end();

      // Random number generator
      std::default_random_engine engine;
      std::uniform_real_distribution<double> distribution(0., 1.);

      Fits::VecRaster<std::complex<double>> mtfSum(broadbandShape);

      // Loop over lambdas
      for (long l = 0; l < lambdas; ++l) {

        // Generate input
        std::generate(pupilBegin, pupilEnd, [&]() {
          return distribution(engine);
        });

        // Perform and time transforms
        chrono.start();
        const auto dft = pupilToPsf.transform().outBuffer(); // Compute the DFT of the pupil function
        chrono.stop();
        psfToMtf.inBuffer().generate(
            [](auto c) {
              return std::norm(c);
            },
            dft); // Feed psfToMft with |pupilToPsf|^2
        chrono.start();
        const auto mtf = psfToMtf.transform().outBuffer(); // Compute the MTF
        chrono.stop();
        for (long y = 0; y < broadbandSide; ++y) {
          for (long x = 0; x < broadbandSide; ++x) {
            mtfSum[{x, y}] += mtf[{x * sideRatio, y * sideRatio}];
          }
        }
      }
      chrono.start();
      psfsToBroadband.transform();
      chrono.stop();
    }
    programChrono.stop();
    logger.info() << "  Done in " << programChrono.last().count() << " ms.";

    // Print aggregate times
    logger.info() << "Parameter-wise timings:";
    for (long i = 0; i < params; ++i) {
      const auto count = dfts[i].chrono.count();
      const auto mean = dfts[i].chrono.mean();
      logger.info() << "  Parameter #" << i << ": " << count << " transforms lasted " << mean << " ms in average.";
    }

    return Elements::ExitCode::OK;
  }
};

MAIN_FOR(EleFourierParallelizationTutorial)

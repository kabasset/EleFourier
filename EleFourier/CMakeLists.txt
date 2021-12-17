CMAKE_MINIMUM_REQUIRED(VERSION 2.8.5)

#===============================================================================
# Load elements_subdir macro here
# Examples:
#   For declaring a project module:
#         elements_subdir(ElementsExamples)
#===============================================================================
elements_subdir(EleFourier)

#===============================================================================
# Load elements_depends_on_subdirs macro here
#   For creating a dependency onto an other accessible module
#         elements_depends_on_subdirs(ElementsKernel)
#===============================================================================
elements_depends_on_subdirs(ElementsKernel)
elements_depends_on_subdirs(EleFits EleFitsUtils EleFitsValidation)

#===============================================================================
# Add the find_package macro (a pure CMake command) here to locate the
# libraries.
# Examples:
#          find_package(CppUnit)
#===============================================================================
find_package(FFTW)

#===============================================================================
# Declare the library dependencies here
# Example:
#         elements_add_library(ElementsExamples src/Lib/*.cpp
#                     INCLUDE_DIRS Boost ElementsKernel
#                     LINK_LIBRARIES Boost ElementsKernel
#                     PUBLIC_HEADERS ElementsExamples)
#===============================================================================
elements_add_library(EleFourier src/lib/*.cpp
                     INCLUDE_DIRS ElementsKernel EleFits EleFitsUtils EleFitsValidation FFTW
                     LINK_LIBRARIES ElementsKernel EleFits EleFitsUtils EleFitsValidation FFTW
                     PUBLIC_HEADERS EleFourier)

#===============================================================================
# Declare the executables here
# Example:
# elements_add_executable(ElementsProgramExample src/Program/ProgramExample.cpp
#                        INCLUDE_DIRS Boost ElementsExamples
#                        LINK_LIBRARIES Boost ElementsExamples)
#===============================================================================
elements_add_executable(EleFourierTutorial src/program/EleFourierTutorial.cpp
                     LINK_LIBRARIES EleFourier)

#===============================================================================
# Declare the Boost tests here
# Example:
# elements_add_unit_test(BoostClassExample tests/src/Boost/ClassExample_test.cpp
#                       EXECUTABLE BoostClassExample_test
#                       INCLUDE_DIRS ElementsExamples
#                       LINK_LIBRARIES ElementsExamples TYPE Boost)
#===============================================================================
elements_add_unit_test(Dft tests/src/Dft_test.cpp 
                     EXECUTABLE EleFourier_Dft_test
                     LINK_LIBRARIES EleFourier
                     TYPE Boost)
elements_add_unit_test(DftPlan tests/src/DftPlan_test.cpp 
                     EXECUTABLE EleFourier_DftPlan_test
                     LINK_LIBRARIES EleFourier
                     TYPE Boost)
elements_add_unit_test(DftType tests/src/DftType_test.cpp 
                     EXECUTABLE EleFourier_DftType_test
                     LINK_LIBRARIES EleFourier
                     TYPE Boost)

#===============================================================================
# Use the following macro for python modules, scripts and aux files:
#  elements_install_python_modules()
#  elements_install_scripts()
#===============================================================================

#===============================================================================
# Declare the Python programs here
# Examples :
# elements_add_python_program(PythonProgramExample
#                             ElementsExamples.PythonProgramExample)
#===============================================================================

#===============================================================================
# Add the elements_install_conf_files macro
# Examples:
#          elements_install_conf_files()
#===============================================================================
elements_install_conf_files()

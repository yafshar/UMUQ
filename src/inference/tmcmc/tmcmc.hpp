#ifndef UMUQ_TMCMC_H
#define UMUQ_TMCMC_H

#include "datatype.hpp"
#include "numerics/function/fitfunction.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "numerics/stats.hpp"
#include "numerics/multimin.hpp"
#include "inference/prior/priordistribution.hpp"
#include "io/io.hpp"
#include "misc/funcallcounter.hpp"
#include "tmcmcstats.hpp"

#include <mutex>
#include <string>
#include <vector>
#include <utility>
#include <functional>
#include <algorithm>

namespace umuq
{

/*!
 * \defgroup Inference_Module Inference module
 * This is the inference module of UMUQ providing all necessary classes of statistical inference in which
 * Bayes' theorem is used to update the probability for a hypothesis as more evidence or information becomes available.
 */

/*!
 * \defgroup TMCMC_Module TMCMC module
 * \ingroup Inference_Module
 *
 * This is the Transitional Markov Chain Monte Carlo Method module of UMUQ providing all necessary classes of this approach.<br>
 * The %UMUQ implementation is an implementation of an unbiased version of Transitional Markov Chain Monte Carlo.<br>
 *
 * Reference: <br>
 * Wu S, et. al. Bayesian Annealed Sequential Importance Sampling: An Unbiased Version <br>
 * of Transitional Markov Chain Monte Carlo. ASME J. Risk Uncertainty Part B. 2017;4(1)
 *
 * The tmcmc class contains additions, adaptations and modifications to the original c implementation
 * of [pi4u](https://github.com/cselab/pi4u) code made available under the [GNU General Public License v2.0](https://www.gnu.org/licenses/gpl-2.0.html) license. <br>
 *
 */

/*! \namespace umuq::tmcmc
 * \ingroup TMCMC_Module
 *
 * \brief Namespace containing all Transitional Markov Chain Monte Carlo Method symbols from the %UMUQ library.
 *
 * Reference: <br>
 * Wu S, et. al. "Bayesian Annealed Sequential Importance Sampling: An Unbiased Version <br>
 * of Transitional Markov Chain Monte Carlo." ASME J. Risk Uncertainty Part B. 2017;4(1)
 */
namespace tmcmc
{

/*! An instance of funcallcounter object */
funcallcounter fc;

/*!
 * \ingroup TMCMC_Module
 *
 * \brief Initialization of MCMC sampling task
 *
 * \tparam FunctionType Function type, which is used in fit function (default \c FITFUN_T<double>)
 *
 * \param TMCMCObj         TMCMC object which is casted to long long
 * \param SamplePoints     Sampling points
 * \param nSamplePoints    Dimension of sample points
 * \param Fvalue           Function values
 * \param nFvalue          Number of function values
 * \param WorkInformation  Information regarding this task work
 */
template <class FunctionType>
void tmcmcInitTask(long long const TMCMCObj, double const *SamplePoints, int const *nSamplePoints, double *Fvalue, int const *nFvalue, int const *WorkInformation);

/*!
 * \ingroup TMCMC_Module
 *
 * \brief Main MCMC sampling task
 *
 * \tparam FunctionType Function type, which is used in fit function (default FITFUN_T<double>)
 *
 * \param TMCMCObj         TMCMC object which is casted to long long
 * \param SamplePoints     Sampling points
 * \param nSamplePoints    Dimension of sample points
 * \param Fvalue           Function values
 * \param nFvalue          Number of function values
 * \param WorkInformation  Information regarding this task work
 * \param nSelection       Number of times this sample being repeated, or number of steps
 * \param PJ               Probability at stage j
 * \param Mean             The proposal PDF for the MCMC step is a Gaussian distribution centered at
 *                         the sample with Mean
 * \param Covariance       The proposal PDF for the MCMC step is a Gaussian distribution with covariance
 *                         equal to \f$ \beta^2 COV \f$
 * \param nBurningSteps    Number of discarding an initial portion of a Markov chain samples
 */
template <class FunctionType>
void tmcmcMainTask(long long const TMCMCObj, double const *SamplePoints, int const *nSamplePoints,
                   double *Fvalue, int const *nFvalue, int const *WorkInformation,
                   double const *PJ, double const *Covariance, int const *nBurningSteps);

/*!
 * \ingroup TMCMC_Module
 *
 * \brief MCMC update sample task
 *
 * \tparam FunctionType Function type, which is used in fit function (default \c FITFUN_T<double>)
 *
 * \param TMCMCObj         TMCMC object which is casted to long long
 * \param SamplePoints     Array of sample points
 * \param nSamplePoints    Dimension of sample points
 * \param Fvalue           Function values at sampling points
 * \param nFvalue          Number of function values
 * \param WorkInformation  Information regarding this task work
 */
template <class FunctionType>
void tmcmcUpdateTask(long long const TMCMCObj, double const *SamplePoints, int const *nSamplePoints, double *Fvalue, int const *nFvalue, int const *WorkInformation);

/*!
 * \ingroup TMCMC_Module
 *
 * \brief TMCMC task type (function pointer)
 *
 * \tparam double     Real data type
 * \tparam FunctionType Function type, which is used in fit function (default \c FITFUN_T<double>)
 */
template <class FunctionType>
using TMCMTASKTYPE = void (*)(long long const, double const *, int const *, double *, int const *, int const *);

/*! True if TMCMC Tasks have been registered, and false otherwise (logical). */
template <class FunctionType>
static bool tmcmcTaskRegistered = false;

} // namespace tmcmc

namespace tmcmc
{

/*! \class tmcmc
 * \ingroup TMCMC_Module
 *
 * \brief This class performs Transitional Markov Chain Monte Carlo Method
 *
 * \tparam FunctionType Function type, which is used in fit function (default FITFUN_T<double>)
 */
template <class FunctionType = FITFUN_T<double>>
class tmcmc
{
public:
  /*!
   * \brief Construct a new tmcmc object
   *
   */
  tmcmc();

  /*!
   * \brief Destroy the tmcmc object
   *
   */
  ~tmcmc();

  /*!
   * \brief Set the Input File Name
   *
   * \param fileName Input file name
   *
   * \return false If the file does not exist
   */
  inline bool setInputFileName(char const *fileName = "");

  /*!
   * \brief Get the Input File Name object
   *
   * \return char const*
   */
  inline std::string getInputFileName();

  /*!
   * \brief Reset the managed object to the correct size, which is read from input file name
   *
   * \param fileName Input file name
   *
   * \return false If it encounters an unexpected problem
   */
  bool reset(char const *fileName = "");

  /*!
   * \brief Set the Fitting Function object to be used
   *
   * \note
   * - std::move is used so that the Fit Function object fitFun may be "moved from",
   * i.e. allowing the efficient transfer of resources from fitFun to tmcmc member object.
   * Thus, the fitFun object is not accessible after calling this function!
   *
   * \param fitFun Fitting Function object
   *
   * \return false If it encounters an unexpected problem
   */
  inline bool setFitFunction(fitFunction<double, FunctionType> &fitFun);

  /*!
   * \brief Set the fitting Function to be used
   *
   * \param Fun  Fitting Function of type (class FunctionType)
   *
   * \return false If it encounters an unexpected problem
   */
  inline bool setFitFunction(FunctionType &Fun);

  /*!
   * \brief Set the fitting Function to be used
   *
   * \param Fun  Fitting Function of type (class FunctionType)
   *
   * \return false If it encounters an unexpected problem
   */
  inline bool setFitFunction(FunctionType const &Fun);

  /*!
   * \brief Setting both the Init Function & fitting Function members of Fit Function member
   *
   * \param InitFun  Initialization function which has the fixed bool type
   * \param Fun      Fitting Function of type (class FunctionType)
   *
   * \return false If it encounters an unexpected problem
   */
  inline bool setFitFunction(std::function<bool()> &InitFun, FunctionType &Fun);

  /*!
   * \brief Setting both the Init Function & fitting Function members of Fit Function member
   *
   * \param InitFun  Initialization function which has the fixed bool type
   * \param Fun      Fitting Function of type (class FunctionType)
   *
   * \return false If it encounters an unexpected problem
   */
  inline bool setFitFunction(std::function<bool()> const &InitFun, FunctionType const &Fun);

  /*!
   * \brief Initialize the algorithm and set up the TORC environemnt
   *
   * \param fileName Input file name
   *
   * \return false If it encounters an unexpected problem
   */
  bool init(char const *fileName = "");

  /*!
   * \brief Check if the restart files are available or not. In case of being available, load, and update the data.
   *
   * \returns true  For restarting from previous generation of sample points
   * \returns false For a fresh start
   */
  inline bool restart();

  /*!
   * \brief Start of the TMCMC algorithm
   *
   * The original formulation of TMCMC constructs multiple intermediate target
   * distributions as follows: <br>
   * \f$ f_j \propto p(D|\theta)^{\zeta_j}p(\theta),~\text{for}~j=1, \cdots, m~\text{and}~\zeta_1 <\zeta_2<\cdots<\zeta_m=1. \f$ <br>
   * Starting from samples drawn from the prior at stage \f$ j = 1. \f$
   *
   * \returns false If it encounters an unexpected problem
   */
  bool iterate0();

  /*!
   * \brief Internal part of the main sampling iteration of the TMCMC algorithm
   *
   * \returns false If it encounters an unexpected problem
   */
  bool iterateInternal();

  /*!
   * \brief Iterative part of the TMCMC algorithm
   *
   * The original formulation of TMCMC constructs multiple intermediate target
   * distributions as follows: <br>
   * \f$ f_j \propto p(D|\theta)^{\zeta_j}p(\theta),~\text{for}~j=1, \cdots,m~\text{and}~\zeta_1 <\zeta_2<\cdots<\zeta_m=1.\f$ <br>
   * Starting from samples drawn from the prior at stage \f$ j = 1, \f$ the
   * posterior samples are obtained by updating the prior samples through
   * \f$ m - 2 \f$ intermediate target distributions.<br>
   * When \f$ \zeta_j = 1, \f$ this procedure stops and the last set of samples
   * is approximately distributed according to the target posterior PDF,
   * \f$~p(\theta | D) \f$. Through the m steps, the evidence is obtained.
   *
   * \returns false  If it encounters an unexpected problem
   */
  bool iterate();

  /*!
   * \brief Process current data, including, calculating the statistics
   * and computing the probabilities based on function values, and draw
   * new samples.
   *
   * Preparing the new data generation includes:<br>
   * - Process the current sampling points data
   * - Find unique sampling points
   * - Calculate the statistics of the data, including acceptance rate
   * - Compute probabilities based on function values for each sample point
   * - Select the new sample points leaders for propagating the chains and update the statistics
   * - Compute number of selections (steps) for each leader chain
   * - Update leaders information from current data information
   *
   * \returns false  If it encounters an unexpected problem
   */
  bool prepareNewGeneration();

public:
  //! Input file name
  std::string inputFilename;

  //! Stream data for getting the problem size and variables from the input file
  stdata Data;

  //! Current data
  database currentData;

  //! Full data
  database fullData;

  //! Experimental data
  database expData;

  //! Running data
  runinfo runData;

  //! Fitting function
  fitFunction<double, FunctionType> fitfun;

private:
  //! Next generation data
  database leadersData;

  //! TMCMC statistical
  tmcmcStats tStats;

public:
  //! Prior distribution object
  priorDistribution<double> prior;

private:
  //! Sample points
  std::vector<double> samplePoints;

  //! Function values
  std::vector<double> fValue;

  //! Array of the work information, it includes : [Generation, Sample, Step]
  int workInformation[3];

  //! Local covariance with the size of [populationSize * sample dimension * sample dimension]
  std::vector<double> localCovariance;

  //! Mutex object
  std::mutex m;
};

template <class FunctionType>
tmcmc<FunctionType>::tmcmc() : inputFilename("input.par")
{
  Torc.reset(nullptr);
}

template <class FunctionType>
tmcmc<FunctionType>::~tmcmc()
{
  if (Torc)
  {
    Torc->TearDown();
  }
}

template <class FunctionType>
inline bool tmcmc<FunctionType>::setInputFileName(char const *fileName)
{
  // Create an instance of the io object
  umuq::io f;
  if (strlen(fileName) > 0)
  {
    inputFilename = std::string(fileName);
    return f.isFileExist(fileName);
  }
  return f.isFileExist(inputFilename);
}

template <class FunctionType>
inline std::string tmcmc<FunctionType>::getInputFileName()
{
  return inputFilename;
}

template <class FunctionType>
bool tmcmc<FunctionType>::reset(char const *fileName)
{
  // Check to see if the input file exists in the PATH or not?
  if (setInputFileName(fileName))
  {
    // Read the input problem size and variables from an input file
    if (Data.load(inputFilename))
    {
      // Creating a database based on the read information
      currentData = std::move(database(Data.nDim, Data.lastPopulationSize));

      // Creating the running information data
      runData = std::move(runinfo(Data.nDim, Data.maxGenerations));

      // Pseudo-random number generator
      psrandom prng(Data.seed);

      // Assign the correct size
      samplePoints.resize(Data.nDim);

      // Function values
      fValue.resize(Data.lastPopulationSize);

      // Local covariance
      if (Data.useLocalCovariance)
      {
        // Local covariance with the size of [populationSize * sample dimension * sample dimension]
        localCovariance.resize(Data.lastPopulationSize * Data.nDim * Data.nDim);
      }
      else
      {
        // Local covariance with the size of [sample dimension * sample dimension]
        localCovariance.resize(Data.nDim * Data.nDim);
      }

      // Initialize the tstats variable from io data
      tStats = std::move(tmcmcStats(Data.options, Data.coefVarPresetThreshold));

      // Construct a prior Distribution object
      prior = std::move(priorDistribution<double>(Data.nDim, Data.priorType));

      // Set the prior parameters
      return prior.set(Data.priorParam1, Data.priorParam2, Data.compositePriorDistribution);
    }
    UMUQFAILRETURN("Failed to initialize the data from the Input file!");
  }
  UMUQFAILRETURN("Input file for the input TMCMC parameter does not exist in the current PATH!!");
}

template <class FunctionType>
bool tmcmc<FunctionType>::setFitFunction(fitFunction<double, FunctionType> &fitFun)
{
  if (fitFun)
  {
    fitfun = std::move(fitFun);
    return true;
  }
  UMUQFAILRETURN("Function is not assigned!");
}

template <class FunctionType>
bool tmcmc<FunctionType>::setFitFunction(FunctionType &Fun)
{
  return fitfun.setFitFunction(Fun);
}

template <class FunctionType>
bool tmcmc<FunctionType>::setFitFunction(FunctionType const &Fun)
{
  return fitfun.setFitFunction(Fun);
}

template <class FunctionType>
bool tmcmc<FunctionType>::setFitFunction(std::function<bool()> &InitFun, FunctionType &Fun)
{
  return fitfun.set(InitFun, Fun);
}

template <class FunctionType>
bool tmcmc<FunctionType>::setFitFunction(std::function<bool()> const &InitFun, FunctionType const &Fun)
{
  return fitfun.set(InitFun, Fun);
}

template <class FunctionType>
bool tmcmc<FunctionType>::init(char const *fileName)
{
  if (fitfun)
  {
    if (setInputFileName(fileName))
    {
      // Initialize the fitting function, this should be done before initializing the TORC environment
      fitfun.init();

      // Create a torc environment object
      Torc.reset(new torcEnvironment);

      // Register tasks
      {
        std::lock_guard<std::mutex> lock(m);

        // Check if TMCMC tasks have not been registered, do it
        if (!tmcmcTaskRegistered<FunctionType>)
        {
          torc_register_task((void *)tmcmcInitTask<FunctionType>);
          torc_register_task((void *)tmcmcMainTask<FunctionType>);
          torc_register_task((void *)tmcmcUpdateTask<FunctionType>);
          tmcmcTaskRegistered<FunctionType> = true;
        }
      }

      // Pseudo-random number generator
      umuq::psrandom prng;

      // Set up the TORC environemnt
      umuq::Torc->SetUp();

      // Set the State of pseudo random number generator
      return prng.setState();
    }
    UMUQFAILRETURN("Input file for the input TMCMC parameter does not exist in the current PATH!!");
  }
  UMUQFAILRETURN("Fitting function is not assigned! \n Fitting function must be set before initializing the TMCMC object!");
}

template <class FunctionType>
inline bool tmcmc<FunctionType>::restart()
{
  // Check if the restart file is available and we can load runData from it
  return runData.load() ? currentData.load("", runData.currentGeneration) : false;
}

template <class FunctionType>
bool tmcmc<FunctionType>::iterate0()
{
  // Check if it is a fresh run, or the restart data is corrupted or is not available
  if (restart())
  {
    return true;
  }

  {
    // Current Generation number
    workInformation[0] = runData.currentGeneration;

    // Step number, this is the starting step
    workInformation[2] = 0;

    // Number of function value (from evaluation) at each sampling point
    int nFvalue = 1;

    // Total number of sampling chains
    int const nChains = Data.eachPopulationSize[0];

    // Loop through all the population size
    for (int i = 0; i < nChains; i++)
    {
      // Sample number
      workInformation[1] = i;

      // Create the input sample points from the prior distribution
      prior.sample(samplePoints);

      // Create and submit tasks
      torc_create(-1, (void (*)())tmcmcInitTask<FunctionType>, 6,
                  1, MPIDatatype<long long>, CALL_BY_REF,
                  Data.nDim, MPIDatatype<double>, CALL_BY_COP,
                  1, MPIDatatype<int>, CALL_BY_COP,
                  1, MPIDatatype<double>, CALL_BY_RES,
                  1, MPIDatatype<int>, CALL_BY_COP,
                  3, MPIDatatype<int>, CALL_BY_COP,
                  reinterpret_cast<long long>(this), samplePoints.data(),
                  &Data.nDim, fValue.data() + i, &nFvalue, workInformation);
    }

    torc_enable_stealing();
    torc_waitall();
    torc_disable_stealing();

    // Count the function calls
    fc.count();

    // Print the summary
    UMUQMSG("server: currentGeneration ", runData.currentGeneration, ": total elapsed time = secs, generation elapsed time = secs for function calls = ", fc.getLocalFunctionCallsNumber());

    // Reset the local function counter to zero
    fc.reset();

    // Check if we should save the mid run data
    if (Data.saveData)
    {
      if (!currentData.save("curgen_db", runData.currentGeneration))
      {
        UMUQFAILRETURN("Failed to write down the current data information!");
      }
    }

    // Running information for checkpoint restart file
    return runData.save();
  }
}

template <class FunctionType>
bool tmcmc<FunctionType>::iterateInternal()
{
  // Current Generation number
  workInformation[0] = runData.currentGeneration;

  // Number of function value (from evaluation) at each sampling point
  int nFvalue = 1;

  // Total number of sampling chains
  int const nChains = leadersData.size();

  // Get the pointer to the sample points
  double *leadersSamplePoints = leadersData.samplePoints.data();

  // Dimension of sample points
  int const nDimSamplePoints = leadersData.nDimSamplePoints;

  // Get this generation probability
  double *PJ = runData.generationProbabilty.data() + runData.currentGeneration;

  // Get the scaled chain covariance as \f$ \beta covariance \f$
  if (!Data.useLocalCovariance)
  {
    std::transform(runData.covariance.begin(), runData.covariance.end(), localCovariance.begin(), [&](double const C) { return Data.bbeta * C; });
  }

  // Number of burning steps
  int const nBurningSteps = 0;

  // Loop through all the population size
  for (int i = 0; i < nChains; i++)
  {
    // Sample number
    workInformation[1] = i;

    // Step number, this is the first step
    workInformation[2] = leadersData.nSelection[i];

    // Fill the input sample points from the leaders
    std::copy(leadersSamplePoints, leadersSamplePoints + nDimSamplePoints, samplePoints.data());
    leadersSamplePoints += nDimSamplePoints;

    if (Data.useLocalCovariance)
    {
    }

    // Create tasks
    torc_create(-1, (void (*)())tmcmcMainTask<FunctionType>, 9,
                1, MPIDatatype<long long>, CALL_BY_REF,
                Data.nDim, MPIDatatype<double>, CALL_BY_COP,
                1, MPIDatatype<int>, CALL_BY_COP,
                1, MPIDatatype<double>, CALL_BY_RES,
                1, MPIDatatype<int>, CALL_BY_COP,
                3, MPIDatatype<int>, CALL_BY_COP,
                1, MPIDatatype<double>, CALL_BY_COP,
                Data.nDim * Data.nDim, MPIDatatype<double>, CALL_BY_COP,
                1, MPIDatatype<int>, CALL_BY_COP,
                reinterpret_cast<long long>(this), samplePoints.data(),
                &Data.nDim, fValue.data() + i, &nFvalue, workInformation,
                PJ, localCovariance.data(), &nBurningSteps);
  }

  torc_enable_stealing();
  torc_waitall();
  torc_disable_stealing();

  // Count the function calls
  fc.count();

  // Print the summary
  UMUQMSG("server: currentGeneration ", runData.currentGeneration, ": total elapsed time = secs, generation elapsed time = secs for function calls = ", fc.getLocalFunctionCallsNumber());

  // Reset the local function counter to zero
  fc.reset();

  // Check if we should save the mid run data
  if (Data.saveData)
  {
    if (!currentData.save("curgen_db", runData.currentGeneration))
    {
      UMUQFAILRETURN("Failed to write down the current data information!");
    }
  }

  // Running information for checkpoint restart
  return runData.save();
}

template <class FunctionType>
bool tmcmc<FunctionType>::iterate()
{
  if (!iterate0())
  {
    UMUQFAILRETURN("Failed to start the TMCMC sampling algorithm!");
  }

  // Prepare new generation from the current information
  if (!prepareNewGeneration())
  {
    UMUQFAILRETURN("Failed to prepare the new generation of sample points!");
  }

  // Broadcast the running information to all of the nodes
  runData.broadcast();

  // Print the sample mean and Sample covariance matrix
  runData.printSampleStatistics();

  // Check the current data probability
  while (runData.generationProbabilty[runData.currentGeneration] < double{1} &&
         runData.currentGeneration < runData.maxGenerations)
  {
    runData.currentGeneration++;

    if (!iterateInternal())
    {
      UMUQFAILRETURN("Failed to update the samples!");
    }

    // Prepare new generation from the current information
    if (!prepareNewGeneration())
    {
      UMUQFAILRETURN("Failed to prepare the new generation of sample points!");
    }

    // Broadcast the running information to all of the nodes
    runData.broadcast();

    // Print the sample mean and Sample covariance matrix
    runData.printSampleStatistics();
  }
}

template <class FunctionType>
bool tmcmc<FunctionType>::prepareNewGeneration()
{
  int const nDimSamplePoints = currentData.nDimSamplePoints;
  int const nCurrentSamplePoints = currentData.size();

  // Total size of the sampling points array Dim * number of sample points
  int const nSize = nCurrentSamplePoints * nDimSamplePoints;

  // Create an instance of the statistics object
  stats s;

#ifdef DEBUG
  // Compute vectors of mean and standard deviation for each dimension
  std::vector<double> mean(nDimSamplePoints);
  std::vector<double> stddev(nDimSamplePoints);

  for (int i = 0; i < nDimSamplePoints; i++)
  {
    mean[i] = s.mean(currentData.samplePoints.data() + i, nSize, nDimSamplePoints);
    stddev[i] = s.stddev(currentData.samplePoints.data() + i, nSize, nDimSamplePoints, mean[i]);
  }

  io f;
  UMUQMSG("Complete data samples on the current Generation Number = ", runData.currentGeneration);
  f.printMatrix("Means", mean.data(), 1, nDimSamplePoints);
  f.printMatrix("Stddev", stddev.data(), 1, nDimSamplePoints);
#endif

  // Now we check to find unique sampling points
  std::vector<double> currentDataUniques(nSize);

  // Get the uniques samples
  s.unique(currentData.samplePoints, nCurrentSamplePoints, nDimSamplePoints, currentDataUniques);

  // Set the number of uniques samples
  runData.setUniqueNumber(currentDataUniques.size());

  {
    // Compute the acceptance rate
    double const acceptanceRate = static_cast<double>(currentDataUniques.size()) / static_cast<double>(nCurrentSamplePoints);

    // Set the acceptance rate
    runData.setAcceptanceRate(acceptanceRate);
  }

#ifdef DEBUG
  for (int i = 0; i < nDimSamplePoints; i++)
  {
    mean[i] = s.mean(currentDataUniques.data() + i, nSize, nDimSamplePoints);
    stddev[i] = s.stddev(currentDataUniques.data() + i, nSize, nDimSamplePoints, mean[i]);
  }

  UMUQMSG("Unique data samples on current Generation Number = ", runData.currentGeneration);
  f.printMatrix("Means", mean.data(), 1, nDimSamplePoints);
  f.printMatrix("Stddev", stddev.data(), 1, nDimSamplePoints);
#endif

  // Create database for leaders selection
  leadersData = std::move(database(nDimSamplePoints, Data.eachPopulationSize[runData.currentGeneration]));

  // Select the new generaion leaders and update the statistics
  if (tStats.selectNewGeneration(Data, currentData, runData, leadersData))
  {
    // Reset the number of selections for each leader chain
    if (leadersData.resetSelection(Data.minChainLength, Data.maxChainLength))
    {
      // Update leaders information from current data
      if (leadersData.updateSelection(currentData))
      {
        // Reset current data number of entries for the new generation of samples
        currentData.resetSize();

#ifdef DEBUG
        // Total number of sampling points
        int const nLeadersSamplePoints = static_cast<int>(leadersData.idxPosition);

        // Total size of the sampling points array Dim * number of sample points
        int const nLeadersSize = nLeadersSamplePoints * nDimSamplePoints;

        for (int i = 0; i < nDimSamplePoints; i++)
        {
          mean[i] = s.mean(leadersData.samplePoints.data() + i, nLeadersSize, nDimSamplePoints);
          stddev[i] = s.stddev(leadersData.samplePoints.data() + i, nLeadersSize, nDimSamplePoints, mean[i]);
        }

        UMUQMSG("Leaders samples = ", runData.currentGeneration);
        f.printMatrix("Means", mean.data(), 1, nDimSamplePoints);
        f.printMatrix("Stddev", stddev.data(), 1, nDimSamplePoints);
#endif

        if (Data.useLocalCovariance)
        {
          // // Total number of sampling points
          // int const nLeadersSamplePoints = static_cast<int>(leadersData.idxPosition);

          // // Assign the correct local covariance size
          // try
          // {
          //   localCovariance.resize(nLeadersSamplePoints * nDimSamplePoints * nDimSamplePoints, double{});
          // }
          // catch (...)
          // {
          //   UMUQFAILRETURN("Failed to allocate memory!");
          // }

          // // Initialize the local covariance to an Identity matrix for each sample
          // for (int i = 0, l = 0; i < nLeadersSamplePoints; i++)
          // {
          //   for (int j = 0; j < nDimSamplePoints; j++)
          //   {
          //     for (int k = 0; k < nDimSamplePoints; k++, l++)
          //     {
          //       if (j == k)
          //       {
          //         localCovariance[l] = double{1};
          //       }
          //     }
          //   }
          // }
          UMUQFAILRETURN("Not implemented yet!");
        }

        return true;
      }
      UMUQFAILRETURN("Failed to update information from the current chain to the leaders!");
    }
    UMUQFAILRETURN("Failed to balance the leaders number of chains selection!");
  }
  UMUQFAILRETURN("Failed to select the leaders for the new generation!");
}

template <class FunctionType>
void tmcmcInitTask(long long const TMCMCObj, double const *SamplePoints, int const *nSamplePoints,
                   double *Fvalue, int const *nFvalue, int const *WorkInformation)
{
  auto tmcmcObj = reinterpret_cast<tmcmc<FunctionType> *>(TMCMCObj);

  // If we have set the fitting function
  if (tmcmcObj->fitfun)
  {
    int const nSamples = *nSamplePoints;
    int const nFunvals = *nFvalue;

    // Create an array of sample points
    std::vector<double> samplePoints{SamplePoints, SamplePoints + nSamples};

    // Create an array of work information
    std::vector<int> workInformation{WorkInformation, WorkInformation + 3};

    // Increment function call counter
    fc.increment();

    // Call the fitting function
    Fvalue[0] = tmcmcObj->fitfun.f(samplePoints.data(), nSamples, Fvalue, nFunvals, workInformation.data());

    // double *fv = Fvalue;
    // fv += (nFunvals > 1 ? 1 : 0);

    // Update the data
    tmcmcObj->currentData.update(samplePoints.data(), Fvalue[0]);

    return;
  }
  UMUQFAIL("Fitting function is not assigned! \n Fitting function must be set before initializing the TMCMC object!");
}

template <class FunctionType>
void tmcmcMainTask(long long const TMCMCObj, double const *SamplePoints, int const *nSamplePoints,
                   double *Fvalue, int const *nFvalue, int const *WorkInformation,
                   double const *PJ, double const *Covariance, int const *nBurningSteps)
{
  auto tmcmcObj = reinterpret_cast<tmcmc<FunctionType> *>(TMCMCObj);

  // If we have set the fitting function
  if (!tmcmcObj->fitfun)
  {
    UMUQFAIL("Fitting function is not assigned! \n Fitting function must be set before initializing the TMCMC object!");
  }

  int const nSamples = *nSamplePoints;
  int const nFunvals = *nFvalue;
  int const nSteps = WorkInformation[2];
  int const nBurnSteps = *nBurningSteps;
  double const pj = *PJ;

  // Create an array of sample points & function values for leaders
  std::vector<double> leaderSamplePoints{SamplePoints, SamplePoints + nSamples};
  std::vector<double> leaderFvalue{Fvalue, Fvalue + nFunvals};

  // Create an array of sample points & function values for candidates
  std::vector<double> candidateSamplePoints(nSamples);
  std::vector<double> candidateFvalue(nFunvals);

  return;

  // Map the data to the Eigen vector format without memory copy
  umuq::EVectorMapType<double> EcandidateSamplePoints(candidateSamplePoints.data(), nSamples);

  // Create an array of work information
  std::vector<int> workInformation{WorkInformation, WorkInformation + 3};

  // Sample mean for the first step
  std::vector<double> chainMean{SamplePoints, SamplePoints + nSamples};

  // Create an instance of the pseudo random number generator
  psrandom prng;

  for (int step = 0; step < nSteps + nBurnSteps; step++)
  {
    if (step > 0)
    {
      std::copy(leaderSamplePoints.begin(), leaderSamplePoints.end(), chainMean.begin());
    }

    umuq::randomdist::multivariateNormalDistribution<double> mvnormal(chainMean.data(), Covariance, nSamples);

    // Generate a candidate
    EcandidateSamplePoints = mvnormal.dist();

    // Check the candidate
    bool isCandidateSamplePointBounded(true);
    for (int i = 0; i < nSamples; i++)
    {
      if (candidateSamplePoints[i] < tmcmcObj->Data.lowerBound[i] ||
          candidateSamplePoints[i] > tmcmcObj->Data.upperBound[i])
      {
        isCandidateSamplePointBounded = false;
        break;
      }
    }

    // If the candidate is in the domain
    if (isCandidateSamplePointBounded)
    {
      // Increment function call counter
      fc.increment();

      // Pass the step number
      workInformation[2] = step;

      /*!
       * \todo
       * Check if the fitting function is in the log mod or not!
       */
      // Call the fitting function
      candidateFvalue[0] = tmcmcObj->fitfun.f(candidateSamplePoints.data(), nSamples,
                                              candidateFvalue.data(), nFunvals, workInformation.data());

      // Accept or Reject

      // The acceptance ratio
      double acceptanceRatio;

      {
        // Calculate the acceptance ratio

        double const candidateLogPrior = tmcmcObj->prior.logpdf(candidateSamplePoints);
        double const leaderLogPrior = tmcmcObj->prior.logpdf(leaderSamplePoints);

        acceptanceRatio = std::exp((candidateLogPrior - leaderLogPrior) + (candidateFvalue[0] - leaderFvalue[0]) * pj);

        if (acceptanceRatio > 1)
        {
          acceptanceRatio = double{1};
        }
      }

      // Generate a uniform random number uniformRandomNumber on [0,1]
      if (prng.unirnd() < acceptanceRatio)
      {
        // Accept the candidate

        // New leader!
        std::copy(candidateSamplePoints.begin(), candidateSamplePoints.end(), leaderSamplePoints.begin());
        std::copy(candidateFvalue.begin(), candidateFvalue.end(), leaderFvalue.begin());

        if (step >= nBurnSteps)
        {
          // Update the data
          tmcmcObj->currentData.update(leaderSamplePoints.data(), leaderFvalue[0]);
        }
      }
      else
      {
        // Reject the candidate

        // Increase counter or add the leader again in the data
        if (step >= nBurnSteps)
        {
          // Update the data
          tmcmcObj->currentData.update(leaderSamplePoints.data(), leaderFvalue[0]);
        }
      }
    }
    else
    {
      // Increase counter or add the leader again in the data
      if (step >= nBurnSteps)
      {
        // Update the data
        tmcmcObj->currentData.update(leaderSamplePoints.data(), leaderFvalue[0]);
      }
    }
  }
  return;
}

template <class FunctionType>
void tmcmcUpdateTask(long long const TMCMCObj, double const *SamplePoints, int const *nSamplePoints, double *Fvalue, int const *nFvalue, int const *WorkInformation)
{
  auto tmcmcObj = reinterpret_cast<tmcmc<FunctionType> *>(TMCMCObj);

  // If we have set the fittiting function
  if (tmcmcObj->fitfun)
  {
    return;
  }
  UMUQFAIL("Fitting function is not assigned! \n Fitting function must be set before initializing the TMCMC object!");
}

} // namespace tmcmc
} // namespace umuq

#endif

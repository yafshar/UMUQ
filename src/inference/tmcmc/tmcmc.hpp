#ifndef UMUQ_TMCMC_H
#define UMUQ_TMCMC_H

#include "data/datatype.hpp"
#include "numerics/function/fitfunction.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "numerics/stats.hpp"
#include "numerics/multimin.hpp"
#include "inference/prior/priordistribution.hpp"
#include "io/io.hpp"
#include "misc/funcallcounter.hpp"
#include "tmcmcstats.hpp"

namespace umuq
{

/*! \namespace tmcmc
 * \brief Namespace containing all the functions for TMCMC algorithm
 *
 */
namespace tmcmc
{

/*!
 * \brief TORC environemnt object
 * 
 * \tparam T Data type
 */
template <typename T>
std::unique_ptr<torcEnvironment<T>> torc;

//! Create an instance of funcallcounter object
funcallcounter fc;

/*!
 * \brief Initialization task
 * 
 * \tparam T  Data type
 * \tparam F  Function type, which is used in fit function (default FITFUN_T<T>) 
 * 
 * \param TMCMCObj         TMCMC object which is casted to long long
 * \param SamplePoints     Sampling points
 * \param nSamplePoints    Dimension of sample points
 * \param Fvalue           Function values
 * \param nFvalue          Number of function values
 * \param WorkInformation  Information regarding this task work 
 */
template <typename T, class F>
void tmcmcInitTask(long long const TMCMCObj, T const *SamplePoints, int const *nSamplePoints, T *Fvalue, int const *nFvalue, int const *WorkInformation);

/*!
 * \brief Main task
 * 
 * \tparam T  Data type
 * \tparam F  Function type, which is used in fit function (default FITFUN_T<T>) 
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
template <typename T, class F>
void tmcmcMainTask(long long const TMCMCObj, T const *SamplePoints, int const *nSamplePoints,
                   T *Fvalue, int const *nFvalue, int const *WorkInformation,
                   T const *PJ, T const *Covariance, int const *nBurningSteps);

/*!
 * \brief 
 * 
 * \tparam T  Data type
 * \tparam F  Function type, which is used in fit function (default FITFUN_T<T>) 
 * 
 * \param TMCMCObj 
 * \param SamplePoints 
 * \param nSamplePoints 
 * \param Fvalue 
 * \param nFvalue 
 * \param WorkInformation 
 */
template <typename T, class F>
void tmcmcUpdateTask(long long const TMCMCObj, T const *SamplePoints, int const *nSamplePoints, T *Fvalue, int const *nFvalue, int const *WorkInformation);

/*!
 * \brief TMCMC task type (function pointer)
 * 
 * \tparam T  Data type
 * \tparam F  Function type, which is used in fit function (default FITFUN_T<T>) 
 */
template <typename T, class F>
using TMCMTASKTYPE = void (*)(long long const, T const *, int const *, T *, int const *, int const *);

//! True if TMCMC Tasks have been registered, and false otherwise (logical).
template <typename T, class F>
static bool tmcmcTaskRegistered = false;

} // namespace tmcmc

namespace tmcmc
{

/*! \class tmcmc
 * \brief 
 * 
 * \tparam T  Data type
 * \tparam F  Function type, which is used in fit function (default FITFUN_T<T>) 
 */
template <typename T, class F = FITFUN_T<T>>
class tmcmc
{
public:
  tmcmc();

  ~tmcmc();

  /*!
   * \brief Set the Input File Name
   * 
   * \param fileName Input file name 
   * 
   * \return true 
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
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  bool reset(char const *fileName = "");

  /*!
   * \brief Set the Fitting Function object to be used
   * 
   * NOTE:
   * std::move is used so that the Fit Function object fitFun may be "moved from", 
   * i.e. allowing the efficient transfer of resources from fitFun to tmcmc member object. 
   * Thus, the fitFun object is not accessible after calling this function!
   * 
   * \param fitFun Fitting Function object 
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  inline bool setFitFunction(fitFunction<T, F> &fitFun);

  /*!
   * \brief Set the fitting Function to be used
   * 
   * \param Fun  Fitting Function of type (class F)
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  inline bool setFitFunction(F &Fun);
  inline bool setFitFunction(F const &Fun);

  /*!
   * \brief Setting both the Init Function & fitting Function members of Fit Function member
   * 
   * \param InitFun  Initialization function which has the fixed bool type
   * \param Fun      Fitting Function of type (class F)
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  inline bool setFitFunction(std::function<bool()> &InitFun, F &Fun);
  inline bool setFitFunction(std::function<bool()> const &InitFun, F const &Fun);

  /*!
   * \brief Initialize the algorithm and set up the TORC environemnt
   * 
   * \param fileName Input file name 
   * 
   * \return true 
   * \return false If it encounters an unexpected problem
   */
  bool init(char const *fileName = "");

  /*!
   * \brief Check if the restart files are available or not. In case of being available, load, and update the data.
   * 
   * \returns true 
   * \returns false If this is a fresh start
   */
  inline bool restart();

  /*!
   * \brief Start of the TMCMC algorithm 
   * 
   * The original formulation of TMCMC constructs multiple intermediate target 
   * distributions as follows:
   * \f[
   *      f_j \propto p(D|\theta)^{\zeta_j}p(\theta) 
   * \f]
   * for \f$ j=1, \cdots,m\f$ and \f$ \zeta_1 <\zeta_2<\cdots<\zeta_m=1.\f$ 
   * Starting from samples drawn from the prior at stage \f$ j = 1. \f$ 
   *  
   * \returns true 
   * \returns false 
   */
  bool iterate0();

  /*!
   * \brief Start of the TMCMC algorithm 
   * 
   *  
   * \returns true 
   * \returns false 
   */
  bool iterate1();

  /*!
   * \brief Iterative part of the TMCMC algorithm 
   * 
   * The original formulation of TMCMC constructs multiple intermediate target 
   * distributions as follows:
   * \f[
   *      f_j \propto p(D|\theta)^{\zeta_j}p(\theta) 
   * \f]
   * for \f$ j=1, \cdots,m\f$ and \f$ \zeta_1 <\zeta_2<\cdots<\zeta_m=1.\f$ 
   * Starting from samples drawn from the prior at stage \f$ j = 1, \f$ the
   * posterior samples are obtained by updating the prior samples through 
   * \f$ m - 2 \f$ intermediate target distributions. 
   * 
   * When \f$ \zeta_j = 1, \f$ this procedure stops and the last set of samples
   * is approximately distributed according to the target posterior PDF, 
   * \f$ p(\theta | D) \f$. Through the m steps, the evidence is obtained. 
   *  
   * \returns true 
   * \returns false 
   */
  bool iterate();

  /*!
   * \brief Process current data, including, calculating the statistics 
   * and computing the probabilities based on function values, and draw 
   * new samples 
   * 
   * Preparing the new data generation includes:
   * - Process the current sampling points data
   * - Find unique sampling points
   * - Calculate the statistics of the data, including acceptance rate
   * - Compute probabilities based on function values for each sample point
   * - Select the new sample points leaders for propagating the chains and update the statistics
   * - Compute number of selections (steps) for each leader chain
   * - Update leaders information from current data information
   * 
   * \returns true 
   * \returns false 
   */
  bool prepareNewGeneration();

public:
  //! Input file name
  std::string inputFilename;

  //! Stream data for getting the problem size and variables from the input file
  stdata<T> Data;

  //! Current data
  database<T> currentData;

  //! Full data
  database<T> fullData;

  //! Experimental data
  database<T> expData;

  //! Running data
  runinfo<T> runData;

  //! Fitting function
  fitFunction<T, F> fitfun;

private:
  //! Next generation data
  database<T> leadersData;

  //! TMCMC statistical
  tmcmcStats<T> tStats;

public:
  //! Prior distribution object
  priorDistribution<T> prior;

  //! Pseudo-random number generator
  psrandom<T> prng;

private:
  //! Sample points
  std::vector<T> samplePoints;

  //! Function values
  std::vector<T> fValue;

  //! Array of the work information, it includes : [Generation, Sample, Step]
  int workInformation[3];

  //! Local covariance with the size of [populationSize * sample dimension * sample dimension]
  std::vector<T> localCovariance;

  //! Mutex object
  std::mutex m;
};

template <typename T, class F>
tmcmc<T, F>::tmcmc() : inputFilename("input.par")
{
  torc<T>.reset(nullptr);
}

template <typename T, class F>
tmcmc<T, F>::~tmcmc()
{
  if (torc<T>)
  {
    torc<T>->TearDown();
  }
}

template <typename T, class F>
inline bool tmcmc<T, F>::setInputFileName(char const *fileName)
{
  //! Create an instance of the io object
  umuq::io f;
  if (strlen(fileName) > 0)
  {
    inputFilename = std::string(fileName);
    return f.isFileExist(fileName);
  }
  return f.isFileExist(inputFilename);
}

template <typename T, class F>
inline std::string tmcmc<T, F>::getInputFileName()
{
  return inputFilename;
}

template <typename T, class F>
bool tmcmc<T, F>::reset(char const *fileName)
{
  //! Check to see if the input file exists in the PATH or not?
  if (setInputFileName(fileName))
  {
    //! Read the input problem size and variables from an input file
    if (Data.load(inputFilename))
    {
      //! Creating a database based on the read information
      currentData = std::move(database<T>(Data.nDim, Data.lastPopulationSize));

      //! Creating the running information data
      runData = std::move(runinfo<T>(Data.nDim, Data.maxGenerations));

      //! Seed the PRNG
      if (!prng.setSeed(Data.seed))
      {
        UMUQWARNING("The Pseudo random number generator has been seeded & initialized before!")
      }

      //! Assign the correct size
      samplePoints.resize(Data.nDim);

      //! Function values
      fValue.resize(Data.lastPopulationSize);

      //! Local covariance
      if (Data.useLocalCovariance)
      {
        //! Local covariance with the size of [populationSize * sample dimension * sample dimension]
        localCovariance.resize(Data.lastPopulationSize * Data.nDim * Data.nDim);
      }
      else
      {
        //! Local covariance with the size of [sample dimension * sample dimension]
        localCovariance.resize(Data.nDim * Data.nDim);
      }

      //! Initialize the tstats variable from io data
      tStats = std::move(tmcmcStats<T>(Data.options));

      //! Construct a prior Distribution object
      prior = std::move(priorDistribution<T>(Data.nDim, Data.priorType));

      //! Set the prior parameters
      return prior.set(Data.priorParam1, Data.priorParam2, Data.compositePriorDistribution);
    }
    UMUQFAILRETURN("Failed to initialize the data from the Input file!");
  }
  UMUQFAILRETURN("Input file for the input TMCMC parameter does not exist in the current PATH!!");
}

template <typename T, class F>
bool tmcmc<T, F>::setFitFunction(fitFunction<T, F> &fitFun)
{
  if (fitFun)
  {
    fitfun = std::move(fitFun);
    return true;
  }
  UMUQFAILRETURN("Function is not assigned!");
}

template <typename T, class F>
bool tmcmc<T, F>::setFitFunction(F &Fun)
{
  return fitfun.setFitFunction(Fun);
}

template <typename T, class F>
bool tmcmc<T, F>::setFitFunction(F const &Fun)
{
  return fitfun.setFitFunction(Fun);
}

template <typename T, class F>
bool tmcmc<T, F>::setFitFunction(std::function<bool()> &InitFun, F &Fun)
{
  return fitfun.set(InitFun, Fun);
}

template <typename T, class F>
bool tmcmc<T, F>::setFitFunction(std::function<bool()> const &InitFun, F const &Fun)
{
  return fitfun.set(InitFun, Fun);
}

template <typename T, class F>
bool tmcmc<T, F>::init(char const *fileName)
{
  if (fitfun)
  {
    if (setInputFileName(fileName))
    {
      //! Initialize the fitting function, this should be done before initializing the TORC environment
      fitfun.init();

      //! Create a torc environment object
      torc<T>.reset(new torcEnvironment<T>);

      //! Register tasks
      {
        std::lock_guard<std::mutex> lock(m);

        // Check if TMCMC tasks have not been registered, do it
        if (!tmcmcTaskRegistered<T, F>)
        {
          torc_register_task((void *)tmcmcInitTask<T, F>);
          torc_register_task((void *)tmcmcMainTask<T, F>);
          torc_register_task((void *)tmcmcUpdateTask<T, F>);
          tmcmcTaskRegistered<T, F> = true;
        }
      }

      //! Set up the TORC environemnt
      torc<T>->SetUp();

      //! Set the State of pseudo random number generator
      if (prng.setState())
      {
        //! Set the Random Number Generator object in the prior
        return prior.setRandomGenerator(&prng);
      }
      UMUQFAILRETURN("Failed to initialize the PRNG or set the state of that!");
    }
    UMUQFAILRETURN("Input file for the input TMCMC parameter does not exist in the current PATH!!");
  }
  UMUQFAILRETURN("Fitting function is not assigned! \n Fitting function must be set before initializing the TMCMC object!");
}

template <typename T, class F>
inline bool tmcmc<T, F>::restart()
{
  //! Check if the restart file is available and we can load runData from it
  return runData.load() ? currentData.load("", runData.currentGeneration) : false;
}

template <typename T, class F>
bool tmcmc<T, F>::iterate0()
{
  //! Check if it is a fresh run, or the restart data is corrupted or is not available
  if (restart())
  {
    return true;
  }

  {
    //! currentGeneration number
    workInformation[0] = runData.currentGeneration;

    //! Step number, this is the first step
    workInformation[2] = 0;

    //! Number of function value at each sampling point
    int nFvalue = 1;

    //! Total number of sampling chains
    int const nChains = Data.eachPopulationSize[0];

    //! Loop through all the population size
    for (int i = 0; i < nChains; i++)
    {
      //! Sample number
      workInformation[1] = i;

      //! Create the input sample points from the prior distribution
      prior.sample(samplePoints);

      //! Create and submit tasks
      torc_create(-1, (void (*)())tmcmcInitTask<T, F>, 6,
                  1, MPIDatatype<long long>, CALL_BY_REF,
                  Data.nDim, MPIDatatype<T>, CALL_BY_COP,
                  1, MPIDatatype<int>, CALL_BY_COP,
                  1, MPIDatatype<T>, CALL_BY_RES,
                  1, MPIDatatype<int>, CALL_BY_COP,
                  3, MPIDatatype<int>, CALL_BY_COP,
                  reinterpret_cast<long long>(this), samplePoints.data(),
                  &Data.nDim, fValue.data() + i, &nFvalue, workInformation);
    }

    torc_enable_stealing();
    torc_waitall();
    torc_disable_stealing();

    //! Count the function calls
    fc.count();

    //! Print the summary
    std::cout << "server: currentGeneration " << runData.currentGeneration << ": total elapsed time = "
              << "secs, generation elapsed time = "
              << "secs for function calls = " << fc.getLocalFunctionCallsNumber() << std::endl;

    //! Reset the local function counter to zero
    fc.reset();

    //! Check if we should save the mid run data
    if (Data.saveData)
    {
      if (!currentData.save("curgen_db", runData.currentGeneration))
      {
        UMUQFAILRETURN("Failed to write down the current data information!");
      }
    }

    //! Running information for checkpoint restart file
    return runData.save();
  }
}

template <typename T, class F>
bool tmcmc<T, F>::iterate1()
{
  //! currentGeneration number
  workInformation[0] = runData.currentGeneration;

  //! Number of function value at each sampling point
  int nFvalue = 1;

  //! Total number of sampling chains
  int const nChains = leadersData.size();

  //! Get the iterator to the sample points
  T *leadersSamplePoints = leadersData.samplePoints.data();

  //! Dimension of sample points
  int const nDimSamplePoints = leadersData.nDimSamplePoints;

  //! Get the
  T *PJ = runData.generationProbabilty.data() + runData.currentGeneration;

  //! Get the chain covariance
  if (!Data.useLocalCovariance)
  {
    std::transform(runData.SS.begin(), runData.SS.end(), localCovariance.begin(), [&](T const C) { return Data.bbeta * C; });
  }

  //! Number of burning steps
  int const nBurningSteps = 0;

  //! Loop through all the population size
  for (int i = 0; i < nChains; i++)
  {
    //! Sample number
    workInformation[1] = i;

    //! Step number, this is the first step
    workInformation[2] = leadersData.nSelection[i];

    //! Fill the input sample points from the leaders
    std::copy(leadersSamplePoints, leadersSamplePoints + nDimSamplePoints, samplePoints.data());
    leadersSamplePoints += nDimSamplePoints;

    if (Data.useLocalCovariance)
    {
    }

    //! Create tasks
    torc_create(-1, (void (*)())tmcmcMainTask<T, F>, 9,
                1, MPIDatatype<long long>, CALL_BY_REF,
                Data.nDim, MPIDatatype<T>, CALL_BY_COP,
                1, MPIDatatype<int>, CALL_BY_COP,
                1, MPIDatatype<T>, CALL_BY_RES,
                1, MPIDatatype<int>, CALL_BY_COP,
                3, MPIDatatype<int>, CALL_BY_COP,
                1, MPIDatatype<T>, CALL_BY_COP,
                Data.nDim * Data.nDim, MPIDatatype<T>, CALL_BY_COP,
                1, MPIDatatype<int>, CALL_BY_COP,
                reinterpret_cast<long long>(this), samplePoints.data(),
                &Data.nDim, fValue.data() + i, &nFvalue, workInformation,
                PJ, localCovariance.data(), &nBurningSteps);
  }

  torc_enable_stealing();
  torc_waitall();
  torc_disable_stealing();

  //! Count the function calls
  fc.count();

  //! Print the summary
  std::cout << "server: currentGeneration " << runData.currentGeneration << ": total elapsed time = "
            << "secs, generation elapsed time = "
            << "secs for function calls = " << fc.getLocalFunctionCallsNumber() << std::endl;

  //! Reset the local function counter to zero
  fc.reset();

  //! Check if we should save the mid run data
  if (Data.saveData)
  {
    if (!currentData.save("curgen_db", runData.currentGeneration))
    {
      UMUQFAILRETURN("Failed to write down the current data information!");
    }
  }

  //! Running information for checkpoint restart
  return runData.save();
}

template <typename T, class F>
bool tmcmc<T, F>::iterate()
{
  if (!iterate0())
  {
    UMUQFAILRETURN("Failed to start the TMCMC sampling algorithm!");
  }

  //! Prepare new generation from the current information
  if (!prepareNewGeneration())
  {
    UMUQFAILRETURN("Failed to prepare the new generation of sample points!");
  }

  //! Broadcast the running information to all of the nodes
  runData.broadcast();

  //! Print the sample mean and Sample covariance matrix
  runData.print();

  //! Check the current data probability
  while (runData.generationProbabilty[runData.currentGeneration] < T{1} && runData.currentGeneration < runData.maxGenerations)
  {
    runData.currentGeneration++;

    if (!iterate1())
    {
      UMUQFAILRETURN("Failed to update the samples!");
    }
  }
}

template <typename T, class F>
bool tmcmc<T, F>::prepareNewGeneration()
{
  int const nDimSamplePoints = currentData.nDimSamplePoints;
  int const nCurrentSamplePoints = currentData.size();

  //! Total size of the sampling points array Dim * number of sample points
  int const nSize = nCurrentSamplePoints * nDimSamplePoints;

  //! Create an instance of the statistics object
  stats s;

#ifdef DEBUG
  //! Compute vectors of mean and standard deviation for each dimension
  std::vector<T> mean(nDimSamplePoints);
  std::vector<T> stddev(nDimSamplePoints);

  for (int i = 0; i < nDimSamplePoints; i++)
  {
    mean[i] = s.mean<T, T>(currentData.samplePoints.data() + i, nSize, nDimSamplePoints);
    stddev[i] = s.stddev<T, T>(currentData.samplePoints.data() + i, nSize, nDimSamplePoints, mean[i]);
  }

  io f;
  std::cout << "Complete data samples on the current Generation Number = " << runData.currentGeneration << std::endl;
  f.printMatrix<T>("Means", mean.data(), 1, nDimSamplePoints);
  f.printMatrix<T>("Stddev", stddev.data(), 1, nDimSamplePoints);
#endif

  //! Now we check to find unique sampling points
  std::vector<T> currentDataUniques(nSize);

  //! Get the uniques samples
  s.unique<T>(currentData.samplePoints, nCurrentSamplePoints, nDimSamplePoints, currentDataUniques);

  //! Set the number of uniques samples
  runData.setUniqueNumber(currentDataUniques.size());

  {
    //! Compute the acceptance rate
    T const acceptanceRate = static_cast<T>(currentDataUniques.size()) / static_cast<T>(nCurrentSamplePoints);

    //! Set the acceptance rate
    runData.setAcceptanceRate(acceptanceRate);
  }

#ifdef DEBUG
  for (int i = 0; i < nDimSamplePoints; i++)
  {
    mean[i] = s.mean<T, T>(currentDataUniques.data() + i, nSize, nDimSamplePoints);
    stddev[i] = s.stddev<T, T>(currentDataUniques.data() + i, nSize, nDimSamplePoints, mean[i]);
  }

  std::cout << "Unique data samples on current Generation Number = " << runData.currentGeneration << std::endl;
  f.printMatrix<T>("Means", mean.data(), 1, nDimSamplePoints);
  f.printMatrix<T>("Stddev", stddev.data(), 1, nDimSamplePoints);
#endif

  //! Create database for leaders selection
  leadersData = std::move(database<T>(nDimSamplePoints, Data.eachPopulationSize[runData.currentGeneration]));

  //! Select the new generaion leaders and update the statistics
  if (tStats.selectNewGeneration(Data, currentData, runData, leadersData))
  {
    //! Reset the number of selections for each leader chain
    if (leadersData.resetSelection(Data.minChainLength, Data.maxChainLength))
    {
      //! Update leaders information from current data
      if (leadersData.updateSelection(currentData))
      {
        //! Reset number of entries
        currentData.idxPosition = 0;

#ifdef DEBUG
        //! Total number of sampling points
        int const nLeadersSamplePoints = static_cast<int>(leadersData.idxPosition);

        //! Total size of the sampling points array Dim * number of sample points
        int const nLeadersSize = nLeadersSamplePoints * nDimSamplePoints;

        for (int i = 0; i < nDimSamplePoints; i++)
        {
          mean[i] = s.mean<T, T>(leadersData.samplePoints.data() + i, nLeadersSize, nDimSamplePoints);
          stddev[i] = s.stddev<T, T>(leadersData.samplePoints.data() + i, nLeadersSize, nDimSamplePoints, mean[i]);
        }

        std::cout << "Leaders samples = " << runData.currentGeneration << std::endl;
        f.printMatrix<T>("Means", mean.data(), 1, nDimSamplePoints);
        f.printMatrix<T>("Stddev", stddev.data(), 1, nDimSamplePoints);
#endif

        if (Data.useLocalCovariance)
        {
          // //! Total number of sampling points
          // int const nLeadersSamplePoints = static_cast<int>(leadersData.idxPosition);

          // //! Assign the correct local covariance size
          // try
          // {
          //   localCovariance.resize(nLeadersSamplePoints * nDimSamplePoints * nDimSamplePoints, T{});
          // }
          // catch (...)
          // {
          //   UMUQFAILRETURN("Failed to allocate memory!");
          // }

          // //! Initialize the local covariance to an Identity matrix for each sample
          // for (int i = 0, l = 0; i < nLeadersSamplePoints; i++)
          // {
          //   for (int j = 0; j < nDimSamplePoints; j++)
          //   {
          //     for (int k = 0; k < nDimSamplePoints; k++, l++)
          //     {
          //       if (j == k)
          //       {
          //         localCovariance[l] = T{1};
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

template <typename T, class F>
void tmcmcInitTask(long long const TMCMCObj, T const *SamplePoints, int const *nSamplePoints,
                   T *Fvalue, int const *nFvalue, int const *WorkInformation)
{
  auto tmcmcObj = reinterpret_cast<tmcmc<T, F> *>(TMCMCObj);

  //! If we have set the fitting function
  if (tmcmcObj->fitfun)
  {
    int const nSamples = *nSamplePoints;
    int const nFunvals = *nFvalue;

    //! Create an array of sample points
    std::vector<T> samplePoints{SamplePoints, SamplePoints + nSamples};

    //! Create an array of work information
    std::vector<int> workInformation{WorkInformation, WorkInformation + 3};

    //! Increment function call counter
    fc.increment();

    //! Call the fitting function
    Fvalue[0] = tmcmcObj->fitfun.f(samplePoints.data(), nSamples, Fvalue, nFunvals, workInformation.data());

    // T *fv = Fvalue;
    // fv += (nFunvals > 1 ? 1 : 0);

    //! Update the data
    tmcmcObj->currentData.update(samplePoints.data(), Fvalue[0]);

    return;
  }
  UMUQFAIL("Fitting function is not assigned! \n Fitting function must be set before initializing the TMCMC object!");
}

template <typename T, class F>
void tmcmcMainTask(long long const TMCMCObj, T const *SamplePoints, int const *nSamplePoints,
                   T *Fvalue, int const *nFvalue, int const *WorkInformation,
                   T const *PJ, T const *Covariance, int const *nBurningSteps)
{
  auto tmcmcObj = reinterpret_cast<tmcmc<T, F> *>(TMCMCObj);

  //! If we have set the fitting function
  if (!tmcmcObj->fitfun)
  {
    UMUQFAIL("Fitting function is not assigned! \n Fitting function must be set before initializing the TMCMC object!");
  }

  int const nSamples = *nSamplePoints;
  int const nFunvals = *nFvalue;
  int const nSteps = WorkInformation[2];
  int const nBurnSteps = *nBurningSteps;
  T const pj = *PJ;

  //! Create an array of sample points & function values for leaders
  std::vector<T> leaderSamplePoints{SamplePoints, SamplePoints + nSamples};
  std::vector<T> leaderFvalue{Fvalue, Fvalue + nFunvals};

  //! Create an array of sample points & function values for candidates
  std::vector<T> candidateSamplePoints(nSamples);
  std::vector<T> candidateFvalue(nFunvals);

  //! Map the data to the Eigen vector format without memory copy
  umuq::EVectorMapType<T> EcandidateSamplePoints(candidateSamplePoints.data(), nSamples);

  //! Create an array of work information
  std::vector<int> workInformation{WorkInformation, WorkInformation + 3};

  //! Sample mean for the first step
  std::vector<T> chainMean{SamplePoints, SamplePoints + nSamples};

  for (int step = 0; step < nSteps + nBurnSteps; step++)
  {
    if (step > 0)
    {
      std::copy(leaderSamplePoints.begin(), leaderSamplePoints.end(), chainMean.begin());
    }

    if (!tmcmcObj->prng.set_mvnormal(chainMean.data(), Covariance, nSamples))
    {
      UMUQFAIL("The pseudo-random number generator failed to set mvnormal object!");
    }

    //! Generate a candidate
    EcandidateSamplePoints = tmcmcObj->prng.mvnormal->dist();

    //! Check the candidate
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

    //! If the candidate is in the domain
    if (isCandidateSamplePointBounded)
    {
      //! Increment function call counter
      fc.increment();

      //! Pass the step number
      workInformation[2] = step;

      /*!
       * TODO:
       * Check if the fitting function is in the log mod or not!
       */
      //! Call the fitting function
      candidateFvalue[0] = tmcmcObj->fitfun.f(candidateSamplePoints.data(), nSamples,
                                              candidateFvalue.data(), nFunvals, workInformation.data());

      //! Accept or Reject

      //! The acceptance ratio
      T acceptanceRatio;

      {
        //! Calculate the acceptance ratio

        T const candidateLogPrior = tmcmcObj->prior.logpdf(candidateSamplePoints);
        T const leaderLogPrior = tmcmcObj->prior.logpdf(leaderSamplePoints);

        acceptanceRatio = std::exp((candidateLogPrior - leaderLogPrior) + (candidateFvalue[0] - leaderFvalue[0]) * pj);

        if (acceptanceRatio > 1)
        {
          acceptanceRatio = T{1};
        }
      }

      //! Generate a uniform random number uniformRandomNumber on [0,1]
      T uniformRandomNumber = tmcmcObj->prng.unirnd();

      if (uniformRandomNumber < acceptanceRatio)
      {
        //! Accept the candidate

        //! New leader!
        std::copy(candidateSamplePoints.begin(), candidateSamplePoints.end(), leaderSamplePoints.begin());
        std::copy(candidateFvalue.begin(), candidateFvalue.end(), leaderFvalue.begin());

        if (step >= nBurnSteps)
        {
          //! Update the data
          tmcmcObj->currentData.update(leaderSamplePoints.data(), leaderFvalue[0]);
        }
      }
      else
      {
        //! Reject the candidate

        //! Increase counter or add the leader again in the data
        if (step >= nBurnSteps)
        {
          //! Update the data
          tmcmcObj->currentData.update(leaderSamplePoints.data(), leaderFvalue[0]);
        }
      }
    }
    else
    {
      //! Increase counter or add the leader again in the data
      if (step >= nBurnSteps)
      {
        //! Update the data
        tmcmcObj->currentData.update(leaderSamplePoints.data(), leaderFvalue[0]);
      }
    }
  }
  return;
}

template <typename T, class F>
void tmcmcUpdateTask(long long const TMCMCObj, T const *SamplePoints, int const *nSamplePoints, T *Fvalue, int const *nFvalue, int const *WorkInformation)
{
  auto tmcmcObj = reinterpret_cast<tmcmc<T, F> *>(TMCMCObj);

  //! If we have set the fittiting function
  if (tmcmcObj->fitfun)
  {
    return;
  }
  UMUQFAIL("Fitting function is not assigned! \n Fitting function must be set before initializing the TMCMC object!");
}

} // namespace tmcmc
} // namespace umuq

#endif

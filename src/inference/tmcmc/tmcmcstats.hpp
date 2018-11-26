#ifndef UMUQ_TMCMCSTATS_H
#define UMUQ_TMCMCSTATS_H

#include "data/datatype.hpp"
#include "numerics/function/fitfunction.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "numerics/stats.hpp"
#include "numerics/multimin.hpp"
#include "inference/prior/priordistribution.hpp"
#include "io/io.hpp"
#include "misc/funcallcounter.hpp"

namespace umuq
{

namespace tmcmc
{

/*! 
 * \ingroup TMCMC_Module
 * 
 * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold.
 * 
 * \tparam DataType       Data type
 * \tparam OutputDataType Output data type (return output result (default is double))
 * 
 * \param  FunValues   An array of log value
 * \param  nFunValues  Number of FunValues 
 * \param  Stride      Element stride 
 * \param  PJ1         \f$ p_{j+1} \f$      
 * \param  PJ          \f$ p_j \f$
 * \param  Tolerance   A prescribed tolerance
 * 
 * \returns OutputDataType The square of the coefficient of variation (COV)
 */
template <typename DataType, typename OutputDataType = double>
OutputDataType CoefVar(DataType const *FunValues, int const nFunValues, int const Stride, DataType const PJ1, DataType const PJ, DataType const Tolerance);

/*! 
 * \ingroup TMCMC_Module
 * 
 * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold.
 * 
 * \tparam DataType       Data type
 * \tparam OutputDataType Output data type (return output result (default is double))
 * 
 * \param  FunValues   An array of log value
 * \param  nFunValues  Number of FunValues 
 * \param  PJ1         \f$ p_{j+1} \f$      
 * \param  PJ          \f$ p_j \f$
 * \param  Tolerance   A prescribed tolerance
 * 
 * \returns OutputDataType The square of the coefficient of variation (COV)
 */
template <typename DataType, typename OutputDataType = double>
OutputDataType CoefVar(DataType const *FunValues, int const nFunValues, DataType const PJ1, DataType const PJ, DataType const Tolerance);

/*! 
 * \ingroup TMCMC_Module
 * 
 * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold.
 * 
 * \tparam DataType       Data type
 * \tparam OutputDataType Output data type (return output result (default is double))
 * 
 * \param  FunValues   An array of log value
 * \param  PJ1         \f$ p_{j+1} \f$      
 * \param  PJ          \f$ p_j \f$
 * \param  Tolerance   A prescribed tolerance
 * 
 * \returns OutputDataType The square of the coefficient of variation (COV)
 */
template <typename DataType, typename OutputDataType = double>
OutputDataType CoefVar(std::vector<DataType> const &FunValues, DataType const PJ1, DataType const PJ, DataType const Tolerance);

/*!
 * \ingroup TMCMC_Module
 * 
 * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold
 * This function is just a wrapper to call CoefVar function but has the proper Function type that can be used 
 * in multidimensional minimization \sa umuq::functionMinimizer.
 * 
 * \tparam DataType Data type 
 * 
 * \param x Input p
 *  
 * \returns The square of the coefficient of variation (COV) for the choice of input p
 */
template <typename DataType>
DataType CoefVarFun(DataType const *x);

/*!
 * \ingroup TMCMC_Module
 * 
 * \brief Pointer to array of function values
 * 
 * \tparam DataType Data type
 */
template <typename DataType>
DataType *functionValues;

/*! Number of function values in functionValues array */
int nFunctionValues;

/*! Current \f$ p_j \f$ */
template <typename DataType>
DataType pj;

/*! A preset threshold for coefficient of variation of the plausibility of weights. */
template <typename DataType>
DataType coefVarPresetThreshold;

} // namespace tmcmc

namespace tmcmc
{

/*! \class tmcmcStats
 * \ingroup TMCMC_Module
 * 
 * \brief Statistic class for TMCMC algorithm
 * 
 * \tparam DataType Data type
 */
template <typename DataType>
class tmcmcStats
{
  public:
    /*!
     * \brief Construct a new tmcmcStats object
     * 
     */
    tmcmcStats();

    /*!
     * \brief Construct a new tmcmc Stats object
     * 
     * \param OptParams               Optimization Parameters
     * \param CoefVarPresetThreshold  A preset threshold for coefficient of variation of the plausibility of weights
     */
    tmcmcStats(optimizationParameters<DataType> const &OptParams, DataType const CoefVarPresetThreshold);

    /*!
     * \brief Move constructor, construct a new tmcmcStats object from an input object
     * 
     * \param other  Input tmcmcStats object
     */
    tmcmcStats(tmcmcStats<DataType> &&other);

    /*!
     * \brief Move assignment operator
     * 
     * \param other 
     * \return tmcmcStats<DataType>& 
     */
    tmcmcStats<DataType> &operator=(tmcmcStats<DataType> &&other);

    /*!
     * \brief Destroy the tmcmc Stats object
     * 
     */
    ~tmcmcStats();

    /*!
     * \brief Set the Random Number Generator object 
     * 
     * \param PRNG  Pseudo-random number object. \sa umuq::random::psrandom.
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    inline bool setRandomGenerator(psrandom<DataType> *PRNG);

    /*!
     * \brief Find the optimum value of  \f$ p_j \f$
     * 
     * The choice of \f$ p_j : j=1,\cdots,m− 1 \f$ is essential.
     * It is desirable to increase the p values slowly so that the transition between adjacent PDFs 
     * is smooth, but if the increase of the p values is too slow, the required number of 
     * intermediate stages will be huge. More intermediate stages mean more computational cost.
     * \f$ p_{j+1} \f$ should be chosen so that the coefficient of variation tolCOV of the plausibility 
     * weights is equal to a prescribed threshold. 
     * 
     * \param FunValues       Function values
     * \param nFunValues      Number of elements in FunValues
     * \param PJ              \f$ p_j \f$
     * \param OptimumP        Optimum value of p
     * \param OptimumCoefVar  Optimum Coef Var
     * 
     * \return true 
     * \return false If it encounters any problem 
     */
    bool findOptimumP(DataType const *FunValues, int const nFunValues, DataType const PJ, DataType *OptimumP, DataType *OptimumCoefVar);

    /*!
     * \brief Find the optimum value of  \f$ p_j \f$ through direct search of the \f$ p \in [0, 4] \f$
     * 
     * 
     * \param FunValues       Function values
     * \param nFunValues      Number of elements in FunValues
     * \param PJ              \f$ p_j \f$
     * \param OptimumP        Optimum value of p
     * \param OptimumCoefVar  Optimum Coef Var
     * 
     * \return true 
     * \return false If it encounters any problem 
     */
    bool searchOptimumP(DataType const *FunValues, int const nFunValues, DataType const PJ, DataType *OptimumP, DataType *OptimumCoefVar);

    /*!
     * \brief This function, prepares and set the new generation from current sample points
     * 
     * \param StreamData   IO data, which includes general information 
     * \param CurrentData  Current generation of sample points
     * \param RunData      Running information data
     * \param Leaders      Leader generation of sample points
     * 
     * \returns true 
     * \returns false If it encounters any problem
     */
    bool selectNewGeneration(stdata<DataType> &StreamData, database<DataType> &CurrentData, runinfo<DataType> &RunData, database<DataType> &Leaders);

  protected:
    /*!
     * \brief Delete a tmcmcStats object copy construction
     * 
     * Make it noncopyable.
     */
    tmcmcStats(tmcmcStats<DataType> const &) = delete;

    /*!
     * \brief Delete a tmcmcStats object assignment
     * 
     * Make it nonassignable
     * 
     * \returns tmcmcStats<DataType>& 
     */
    tmcmcStats<DataType> &operator=(tmcmcStats<DataType> const &) = delete;

  private:
    /*! Optimizer information */
    optimizationParameters<DataType> optParams;

    /*! Create an instance of the function minimizer */
    std::unique_ptr<umuq::functionMinimizer<DataType>> fMinimizer;

    /*! pseudo-random numbers */
    psrandom<DataType> *prng;
};

template <typename DataType>
tmcmcStats<DataType>::tmcmcStats() : optParams(),
                                     fMinimizer(new umuq::simplexNM2<DataType>),
                                     prng(nullptr)
{
    // First we have to set the minimizer dimension
    if (!fMinimizer->reset(1))
    {
        UMUQFAIL("Failed to set the minimizer dimension!");
    }

    // Set the preset threshold for coefficient of variation of the plausibility of weights to the default value of 1
    coefVarPresetThreshold<DataType> = DataType{1};
}

template <typename DataType>
tmcmcStats<DataType>::tmcmcStats(optimizationParameters<DataType> const &OptParams, DataType const CoefVarPresetThreshold) : optParams(OptParams),
                                                                                                                             prng(nullptr)
{
    // Get the correct instance of the minimizer
    switch (optParams.FunctionMinimizerType)
    {
    case (FunctionMinimizerTypes::SIMPLEXNM):
        fMinimizer.reset(new umuq::simplexNM<DataType>);
        break;
    case (FunctionMinimizerTypes::SIMPLEXNM2):
        fMinimizer.reset(new umuq::simplexNM2<DataType>);
        break;
    case (FunctionMinimizerTypes::SIMPLEXNM2RND):
        fMinimizer.reset(new umuq::simplexNM2Rnd<DataType>);
        break;
    default:
        UMUQFAIL("Unknown function minimizer type!");
        break;
    }

    // First we have to set the minimizer dimension
    if (!fMinimizer->reset(1))
    {
        UMUQFAIL("Failed to set the minimizer dimension!");
    }

    // Set the preset threshold for coefficient of variation of the plausibility of weights
    coefVarPresetThreshold<DataType> = CoefVarPresetThreshold;
}

template <typename DataType>
tmcmcStats<DataType>::tmcmcStats(tmcmcStats<DataType> &&other) : optParams(std::move(other.optParams)),
                                                                 fMinimizer(std::move(other.fMinimizer)),
                                                                 prng(other.prng) {}

template <typename DataType>
tmcmcStats<DataType> &tmcmcStats<DataType>::operator=(tmcmcStats<DataType> &&other)
{
    optParams = std::move(other.optParams);
    fMinimizer = std::move(other.fMinimizer);
    prng = other.prng;
    return *this;
}

template <typename DataType>
tmcmcStats<DataType>::~tmcmcStats(){};

template <typename DataType>
inline bool tmcmcStats<DataType>::setRandomGenerator(psrandom<DataType> *PRNG)
{
    if (PRNG)
    {
        if (PRNG_initialized)
        {
            prng = PRNG;
            return true;
        }
        UMUQFAILRETURN("One should set the state of the pseudo random number generator before setting it to this distribution!");
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
}

template <typename DataType>
bool tmcmcStats<DataType>::findOptimumP(DataType const *FunValues, int const nFunValues, DataType const PJ, DataType *OptimumP, DataType *OptimumCoefVar)
{
    // Set the global helper pointer & variables
    functionValues<DataType> = FunValues;
    nFunctionValues = nFunValues;
    pj<DataType> = PJ;

    // Second, we have to set the function, input vector and stepsize in the function minimizer
    if (!fMinimizer->set(CoefVarFun<DataType>, &pj<DataType>, &optParams.Step))
    {
        UMUQFAILRETURN("Failed to set the minimizer!");
    }

    // Third, initialize the minimizer
    if (!fMinimizer->init())
    {
        UMUQFAILRETURN("Failed to initialize the minimizer!");
    }

    // Print the initial starting points, if it is requested
    if (optParams.Display)
    {
        DataType *x = fMinimizer->getX();
        std::cout << "x =" << x[0] << std::endl;
    }

    // Forth, iterate until we reach the absolute tolerance

    // Fail:-1, Success:0, Continue:1
    int status = 1;

    int iter(0);
    while (iter < optParams.MaxIter && status == 1)
    {
        iter++;

        if (!fMinimizer->iterate())
        {
            UMUQFAILRETURN("Failed to iterate the minimizer!");
        }

        if (optParams.Display)
        {
            DataType *x = fMinimizer->getX();
            std::cout << iter << ": ";
            std::cout << "CoefVar(p=" << x[0] << ") =" << fMinimizer->getMin() << ", & characteristic size =" << fMinimizer->size() << std::endl;
        }

        status = fMinimizer->testSize(optParams.Tolerance);
    }

    // If it did not fail
    if (status == 0 || status == 1)
    {
        OptimumP = fMinimizer->getX();
        *OptimumCoefVar = fMinimizer->getMin();

        if (optParams.Display)
        {
            std::cout << fMinimizer->getName() << ", on CoefVarFun : " << iter << " iters, CoefVar(p)=" << *OptimumCoefVar << std::endl;
            std::cout << ((status == 0) ? "Converged to minimum at p = " : "Stopped at p = ");
            std::cout << *OptimumP << std::endl;
        }
        return (status ? (std::abs(fMinimizer->getMin()) <= optParams.Tolerance) : true);
    }
    UMUQFAILRETURN("Could not find the optimum value of p through optimization!");
}

template <typename DataType>
bool tmcmcStats<DataType>::searchOptimumP(DataType const *FunValues, int const nFunValues, DataType const PJ, DataType *OptimumP, DataType *OptimumCoefVar)
{
    DataType MinValp = DataType{};
    DataType MaxValp = DataType{4};

    // Initialize the step size to the prescribed user value
    DataType stepSize(optParams.Step);

    for (;;)
    {
        if (optParams.Display)
        {
            std::cout << "Search for optimum p in [" << MinValp << ", " << MaxValp << "] with step size =" << stepSize << std::endl;
        }

        // Define some variables to use in the search
        bool found(false);

        // Maximum number of iterations
        std::size_t const MaxIter = static_cast<std::size_t>((MaxValp - MinValp) / stepSize);

        // Optimum value of p
        DataType optimumP(MinValp);

        // Max value of CoefVar
        DataType MaxCoefVar(std::numeric_limits<DataType>::max());

        // We search the space for the optimum value
        for (std::size_t iter = 0; iter < MaxIter; iter++)
        {
            DataType const p = MinValp + iter * stepSize;
            DataType const NewCoefVar = CoefVarFun<DataType>(&p);

            if (NewCoefVar < MaxCoefVar)
            {
                MaxCoefVar = NewCoefVar;
                optimumP = p;

                if (MaxCoefVar <= coefVarPresetThreshold<DataType>)
                {
                    found = true;
                    break;
                }
            }
        }

        // Success
        if (found)
        {
            *OptimumP = optimumP;
            *OptimumCoefVar = MaxCoefVar;

            if (optParams.Display)
            {
                std::cout << "Search on CoefVarFun : CoefVar(p)=" << *OptimumCoefVar << std::endl;
                std::cout << "Converged to minimum at p = " << *OptimumP << std::endl;
            }
            return true;
        }
        else
        {
            // If optimumP is not within Tolerance, we can go back and retry with better refinement (more iterations)

            MinValp = optimumP - 10 * stepSize;
            if (MinValp < 0)
            {
                MinValp = DataType{};
            }

            MaxValp = optimumP + 10 * stepSize;
            if (MaxValp > DataType{4})
            {
                MaxValp = DataType{4};
            }

            stepSize /= DataType{10};
            if (stepSize < machinePrecision<DataType>)
            {
                // Fail:-1
                UMUQFAILRETURN("Could not find the optimum value of p through search!");
            }
        }
    }
    UMUQFAILRETURN("Could not find the optimum value of p through search!");
}

template <typename DataType>
bool tmcmcStats<DataType>::selectNewGeneration(stdata<DataType> &StreamData, database<DataType> &CurrentData, runinfo<DataType> &RunData, database<DataType> &Leaders)
{
    // current generation
    int const currentGeneration = RunData.currentGeneration;

    // Next generation
    int const newGeneration = currentGeneration + 1;

    // Probabilty at each generation
    DataType *generationProbabilty = RunData.generationProbabilty.data();

    // Get the pointer to the function values
    DataType *Fvalue = CurrentData.fValue.data();

    // Total number of function values
    int const nCurrentSamplePoints = CurrentData.size();

    {
        DataType optimumP(DataType{});
        DataType optimumCoefVar(DataType{});

        // Probability of the current generation
        DataType const PJ = generationProbabilty[currentGeneration];

        bool status(true);

        // Find the optimum value of \f$ p_{j+1} \f$ through optimization
        if (!findOptimumP(Fvalue, nCurrentSamplePoints, PJ, &optimumP, &optimumCoefVar))
        {
            // In case we did not find the optimum value through optimization, we do the direct search
            status = searchOptimumP(Fvalue, nCurrentSamplePoints, PJ, &optimumP, &optimumCoefVar);
        }

        // If we find the optimum value a
        if (status && optimumP > PJ)
        {
            generationProbabilty[newGeneration] = optimumP;
            RunData.CoefVar[newGeneration] = optimumCoefVar;
        }
        else
        {
            // Increase the probability just a very small value
            generationProbabilty[newGeneration] = PJ + 0.1 * optParams.Step;
            RunData.CoefVar[newGeneration] = RunData.CoefVar[currentGeneration];
        }
    }

    // If the probability of this generation is greater than one, then it is the last step
    if (generationProbabilty[newGeneration] > 1)
    {
        generationProbabilty[newGeneration] = DataType{1};
        StreamData.eachPopulationSize[newGeneration] = StreamData.lastPopulationSize;
    }

    std::vector<DataType> weight(nCurrentSamplePoints);

    {
        // Get the PJ1 - PJ
        DataType const probabiltyDiff = generationProbabilty[newGeneration] - generationProbabilty[currentGeneration];

        // The function value is in log space
        for (int i = 0; i < nCurrentSamplePoints; i++)
        {
            weight[i] = Fvalue[i] * probabiltyDiff;
        }
    }

    stats s;
    DataType const weightMax = s.maxelement<DataType, DataType>(weight);

    std::for_each(weight.begin(), weight.end(), [&](DataType &w_i) { w_i = std::exp(w_i - weightMax); });

    if (optParams.Display)
    {
        io f;
        f.printMatrix<DataType>("Weight matrix", weight.data(), 1, nCurrentSamplePoints);
    }

    // Compute the weight sum
    DataType const weightSum = s.sum<DataType, DataType>(weight);

    // Normalize the weight
    std::for_each(weight.begin(), weight.end(), [&](DataType &w_i) { w_i /= weightSum; });

    if (optParams.Display)
    {
        io f;
        f.printMatrix<DataType>("Normalized Weight matrix", weight.data(), 1, nCurrentSamplePoints);
    }

    {
        // Keep the value for computing evidence
        RunData.logselection[currentGeneration] = std::log(weightSum) + weightMax - std::log(static_cast<DataType>(nCurrentSamplePoints));

        if (optParams.Display)
        {
            io f;
            f.printMatrix<DataType>("log selection matrix", RunData.logselection.data(), 1, newGeneration);
        }
    }

    // Compute the current Generation CoefVar
    RunData.CoefVar[currentGeneration] = s.coefvar<DataType, DataType>(weight);

    if (optParams.Display)
    {
        io f;
        f.printMatrix<DataType>("CoefVar matrix", RunData.CoefVar.data(), 1, newGeneration);
    }

    if (prng)
    {
        // Dimension of the sampling points
        int const nDimSamplePoints = CurrentData.nDimSamplePoints;

        {
            if (Leaders.nDimSamplePoints != nDimSamplePoints)
            {
                UMUQFAILRETURN("Sampling data dimension does not match with the leaders database!");
            }

            if (Leaders.nSamplePoints < nCurrentSamplePoints)
            {
                UMUQFAILRETURN("Leader database size is not large enough!");
            }

            // Get the total number of samples at the current generation
            int const nSamples = StreamData.eachPopulationSize[currentGeneration];

            // Get the pointer
            int *nSelection = Leaders.nSelection.data();

            // Get the number of selection based on the probability
            prng->multinomial(weight.data(), nCurrentSamplePoints, nSamples, nSelection);

            if (optParams.Display)
            {
                io f;
                f.printMatrix<int>("Number of selection = [", nSelection, 1, nCurrentSamplePoints, "]\n----------------------------------------\n");
            }
        }

        // Total size of the sampling points array
        int const nSize = nCurrentSamplePoints * nDimSamplePoints;

        // Compute vectors of mean and standard deviation for each dimension
        std::vector<DataType> thetaMean(nDimSamplePoints);

        for (int i = 0; i < nDimSamplePoints; i++)
        {
            arrayWrapper<DataType> Parray(CurrentData.samplePoints.data() + i, nSize, nDimSamplePoints);
            thetaMean[i] = std::inner_product(weight.begin(), weight.end(), Parray.begin(), DataType{});
        }

        std::copy(thetaMean.begin(), thetaMean.end(), RunData.meantheta.data() + currentGeneration * nDimSamplePoints);

        if (optParams.Display)
        {
            io f;
            f.printMatrix<int>("Mean of theta = [", thetaMean.data(), 1, nDimSamplePoints, "]\n----------------------------------------\n");
        }

        {
            DataType *Covariance = RunData.covariance.data();

            for (int i = 0; i < nDimSamplePoints; i++)
            {
                arrayWrapper<DataType> iArray(CurrentData.samplePoints.data() + i, nSize, nDimSamplePoints);

                DataType const iMean = thetaMean[i];

                for (int j = i; j < nDimSamplePoints; j++)
                {
                    arrayWrapper<DataType> jArray(CurrentData.samplePoints.data() + j, nSize, nDimSamplePoints);

                    DataType const jMean = thetaMean[j];

                    DataType S(0);
                    for (auto i = iArray.begin(), j = jArray.begin(), k = weight.begin(); i != iArray.end(); i++, j++, k++)
                    {
                        DataType const d1 = *i - iMean;
                        DataType const d2 = *j - jMean;
                        S += *k * d1 * d2;
                    }
                    Covariance[i * nDimSamplePoints + j] = S;
                }
            }

            for (int i = 0; i < nDimSamplePoints; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    Covariance[i * nDimSamplePoints + j] = Covariance[j * nDimSamplePoints + i];
                }
            }

            // If the covariance matrix is not positive definite
            if (!isSelfAdjointMatrixPositiveDefinite<DataType>(Covariance, nDimSamplePoints))
            {
                // We force it to be positive definite
                forceSelfAdjointMatrixPositiveDefinite<DataType>(Covariance, nDimSamplePoints);
            }

            if (optParams.Display)
            {
                io f;
                f.printMatrix<int>("Covariance = [", Covariance, nDimSamplePoints, nDimSamplePoints, "]\n----------------------------------------\n");
            }
        }
        return true;
    }
    UMUQFAILRETURN("The pseudo-random number generator object is not assigned!");
}

template <typename DataType, typename OutputDataType>
OutputDataType CoefVar(DataType const *FunValues, int const nFunValues, int const Stride, DataType const PJ1, DataType const PJ, DataType const Tolerance)
{
    arrayWrapper<DataType> iArray(FunValues, nFunValues, Stride);

    std::vector<OutputDataType> weight(iArray.size());

    stats s;

    {
        //Find the maximum value in the array of FunValues of size nFunValues
        DataType const fMaxValue = s.maxelement<DataType>(iArray);

        OutputDataType const diff = static_cast<OutputDataType>(PJ1 - PJ);

        //Compute the weight
        for (std::size_t i = 0; i < iArray.size(); i++)
        {
            weight[i] = std::exp((iArray[i] - fMaxValue) * diff);
        }
    }

    { //Compute the summation of weight
        OutputDataType const weightsum = s.sum<OutputDataType, OutputDataType>(weight);

        //Normalize the weight
        std::for_each(weight.begin(), weight.end(), [&](OutputDataType &w) { w /= weightsum; });
    }

    //Compute the mean
    OutputDataType const weightmean = s.mean<OutputDataType, OutputDataType>(weight);

    //Compute the standard deviation
    OutputDataType const weightstddev = s.stddev<OutputDataType, OutputDataType>(weight, weightmean);

    //return the square of the coefficient of variation (COV)
    return std::pow(weightstddev / weightmean - Tolerance, 2);
}

template <typename DataType, typename OutputDataType>
OutputDataType CoefVar(DataType const *FunValues, int const nFunValues, DataType const PJ1, DataType const PJ, DataType const Tolerance)
{
    std::vector<OutputDataType> weight(nFunValues);

    stats s;

    {
        //Find the maximum value in the array of FunValues of size nFunValues
        DataType const fMaxValue = s.maxelement<DataType>(FunValues, nFunValues);

        OutputDataType const diff = static_cast<OutputDataType>(PJ1 - PJ);

        //Compute the weight
        for (int i = 0; i < nFunValues; i++)
        {
            weight[i] = std::exp((FunValues[i] - fMaxValue) * diff);
        }
    }

    { //Compute the summation of weight
        OutputDataType const weightsum = s.sum<OutputDataType, OutputDataType>(weight);

        //Normalize the weight
        std::for_each(weight.begin(), weight.end(), [&](OutputDataType &w) { w /= weightsum; });
    }

    //Compute the mean
    OutputDataType const weightmean = s.mean<OutputDataType, OutputDataType>(weight);

    //Compute the standard deviation
    OutputDataType const weightstddev = s.stddev<OutputDataType, OutputDataType>(weight, weightmean);

    //return the square of the coefficient of variation (COV)
    return std::pow(weightstddev / weightmean - Tolerance, 2);
}

template <typename DataType, typename OutputDataType>
OutputDataType CoefVar(std::vector<DataType> const &FunValues, DataType const PJ1, DataType const PJ, DataType const Tolerance)
{
    std::vector<OutputDataType> weight(FunValues.size());

    stats s;

    {
        //Find the maximum value in the array of FunValues of size nFunValues
        DataType const fMaxValue = s.maxelement<DataType>(FunValues);

        OutputDataType const diff = static_cast<OutputDataType>(PJ1 - PJ);

        //Compute the weight
        for (std::size_t i = 0; i < FunValues.size(); i++)
        {
            weight[i] = std::exp((FunValues[i] - fMaxValue) * diff);
        }
    }

    { //Compute the summation of weight
        OutputDataType const weightsum = s.sum<OutputDataType, OutputDataType>(weight);

        //Normalize the weight
        std::for_each(weight.begin(), weight.end(), [&](OutputDataType &w) { w /= weightsum; });
    }

    //Compute the mean
    OutputDataType const weightmean = s.mean<OutputDataType, OutputDataType>(weight);

    //Compute the standard deviation
    OutputDataType const weightstddev = s.stddev<OutputDataType, OutputDataType>(weight, weightmean);

    //return the square of the coefficient of variation (COV)
    return std::pow(weightstddev / weightmean - Tolerance, 2);
}

template <typename DataType>
DataType CoefVarFun(DataType const *x)
{
    return CoefVar<DataType, DataType>(functionValues<DataType>, nFunctionValues, *x, pj<DataType>, coefVarPresetThreshold<DataType>);
}

} // namespace tmcmc
} // namespace umuq

#endif // UMUQ_TMCMCSTATS

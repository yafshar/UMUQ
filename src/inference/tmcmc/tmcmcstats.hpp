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

/*! \fn CoefVar
 * \ingroup TMCMC_Module
 * 
 * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold.
 * 
 * \tparam T    Input data type
 * \tparam TOut Output data type (return output result (default is double))
 * 
 * \param  FunValues   An array of log value
 * \param  nFunValues  Number of FunValues 
 * \param  Stride      Element stride 
 * \param  PJ1         \f$ p_{j+1} \f$      
 * \param  PJ          \f$ p_j \f$
 * \param  Tolerance   A prescribed tolerance
 * 
 * \returns the square of the coefficient of variation (COV)
 */
template <typename T, typename TOut = double>
TOut CoefVar(T const *FunValues, int const nFunValues, int const Stride, T const PJ1, T const PJ, T const Tolerance);

template <typename T, typename TOut = double>
TOut CoefVar(T const *FunValues, int const nFunValues, T const PJ1, T const PJ, T const Tolerance);

template <typename T, typename TOut = double>
TOut CoefVar(std::vector<T> const &FunValues, T const PJ1, T const PJ, T const Tolerance);

/*! \fn CoefVarFun
 * \ingroup TMCMC_Module
 * 
 * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold
 * This function is just a wrapper to call CoefVar function but has the proper Function type that can be used 
 * in multidimensional minimization \sa functionMinimizer.
 * 
 * \tparam T Data type 
 * 
 * \param x Input p
 *  
 * \returns The square of the coefficient of variation (COV) for the choice of input p
 */
template <typename T>
T CoefVarFun(T const *x);

/*!
 * \ingroup TMCMC_Module
 * 
 * \brief Pointer to array of function values
 * 
 * \tparam T Data type
 */
template <typename T>
T *functionValues;

//! Number of function values in functionValues array
int nFunctionValues;

//! Current \f$ p_j \f$
template <typename T>
T pj;

//! A preset threshold for coefficient of variation of the plausibility of weights.
template <typename T>
T coefVarPresetThreshold;

} // namespace tmcmc

namespace tmcmc
{

/*! \class tmcmcStats
 * \ingroup TMCMC_Module
 * 
 * \brief Statistic class for TMCMC algorithm
 * 
 * \tparam T Data type
 */
template <typename T>
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
    tmcmcStats(optimizationParameters<T> const &OptParams, T const CoefVarPresetThreshold);

    /*!
     * \brief Move constructor, construct a new tmcmcStats object from an input object
     * 
     * \param other  Input tmcmcStats object
     */
    tmcmcStats(tmcmcStats<T> &&other);

    /*!
     * \brief Move assignment operator
     * 
     * \param other 
     * \return tmcmcStats<T>& 
     */
    tmcmcStats<T> &operator=(tmcmcStats<T> &&other);

    /*!
     * \brief Destroy the tmcmc Stats object
     * 
     */
    ~tmcmcStats();

    /*!
     * \brief Set the Random Number Generator object 
     * 
     * \param PRNG  Pseudo-random number object \sa psrandom
     * 
     * \return true 
     * \return false If it encounters an unexpected problem
     */
    inline bool setRandomGenerator(psrandom<T> *PRNG);

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
    bool findOptimumP(T const *FunValues, int const nFunValues, T const PJ, T *OptimumP, T *OptimumCoefVar);

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
    bool searchOptimumP(T const *FunValues, int const nFunValues, T const PJ, T *OptimumP, T *OptimumCoefVar);

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
    bool selectNewGeneration(stdata<T> &StreamData, database<T> &CurrentData, runinfo<T> &RunData, database<T> &Leaders);

  private:
    //! Make it noncopyable
    tmcmcStats(tmcmcStats<T> const &) = delete;

    //! Make it not assignable
    tmcmcStats<T> &operator=(tmcmcStats<T> const &) = delete;

  private:
    //! Optimizer information
    optimizationParameters<T> optParams;

    //! Create an instance of the function minimizer
    std::unique_ptr<umuq::functionMinimizer<T>> fMinimizer;

    //! pseudo-random numbers
    psrandom<T> *prng;
};

template <typename T>
tmcmcStats<T>::tmcmcStats() : optParams(),
                              fMinimizer(new umuq::simplexNM2<T>),
                              prng(nullptr)
{
    // First we have to set the minimizer dimension
    if (!fMinimizer->reset(1))
    {
        UMUQFAIL("Failed to set the minimizer dimension!");
    }

    // Set the preset threshold for coefficient of variation of the plausibility of weights to the default value of 1
    coefVarPresetThreshold<T> = T{1};
}

template <typename T>
tmcmcStats<T>::tmcmcStats(optimizationParameters<T> const &OptParams, T const CoefVarPresetThreshold) : optParams(OptParams),
                                                                                                        prng(nullptr)
{
    // Get the correct instance of the minimizer
    switch (optParams.FunctionMinimizerType)
    {
    case (FunctionMinimizerTypes::SIMPLEXNM):
        fMinimizer.reset(new umuq::simplexNM<T>);
        break;
    case (FunctionMinimizerTypes::SIMPLEXNM2):
        fMinimizer.reset(new umuq::simplexNM2<T>);
        break;
    case (FunctionMinimizerTypes::SIMPLEXNM2RND):
        fMinimizer.reset(new umuq::simplexNM2Rnd<T>);
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
    coefVarPresetThreshold<T> = CoefVarPresetThreshold;
}

template <typename T>
tmcmcStats<T>::tmcmcStats(tmcmcStats<T> &&other) : optParams(std::move(other.optParams)),
                                                   fMinimizer(std::move(other.fMinimizer)),
                                                   prng(other.prng) {}

template <typename T>
tmcmcStats<T> &tmcmcStats<T>::operator=(tmcmcStats<T> &&other)
{
    optParams = std::move(other.optParams);
    fMinimizer = std::move(other.fMinimizer);
    prng = other.prng;
    return *this;
}

template <typename T>
tmcmcStats<T>::~tmcmcStats(){};

template <typename T>
inline bool tmcmcStats<T>::setRandomGenerator(psrandom<T> *PRNG)
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

template <typename T>
bool tmcmcStats<T>::findOptimumP(T const *FunValues, int const nFunValues, T const PJ, T *OptimumP, T *OptimumCoefVar)
{
    // Set the global helper pointer & variables
    functionValues<T> = FunValues;
    nFunctionValues = nFunValues;
    pj<T> = PJ;

    // Second, we have to set the function, input vector and stepsize in the function minimizer
    if (!fMinimizer->set(CoefVarFun<T>, &pj<T>, &optParams.Step))
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
        T *x = fMinimizer->getX();
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
            T *x = fMinimizer->getX();
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

template <typename T>
bool tmcmcStats<T>::searchOptimumP(T const *FunValues, int const nFunValues, T const PJ, T *OptimumP, T *OptimumCoefVar)
{
    T MinValp = T{};
    T MaxValp = T{4};

    // Initialize the step size to the prescribed user value
    T stepSize(optParams.Step);

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
        T optimumP(MinValp);

        // Max value of CoefVar
        T MaxCoefVar(std::numeric_limits<T>::max());

        // We search the space for the optimum value
        for (std::size_t iter = 0; iter < MaxIter; iter++)
        {
            T const p = MinValp + iter * stepSize;
            T const NewCoefVar = CoefVarFun<T>(&p);

            if (NewCoefVar < MaxCoefVar)
            {
                MaxCoefVar = NewCoefVar;
                optimumP = p;

                if (MaxCoefVar <= coefVarPresetThreshold<T>)
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
                MinValp = T{};
            }

            MaxValp = optimumP + 10 * stepSize;
            if (MaxValp > T{4})
            {
                MaxValp = T{4};
            }

            stepSize /= T{10};
            if (stepSize < machinePrecision<T>)
            {
                // Fail:-1
                UMUQFAILRETURN("Could not find the optimum value of p through search!");
            }
        }
    }
    UMUQFAILRETURN("Could not find the optimum value of p through search!");
}

template <typename T>
bool tmcmcStats<T>::selectNewGeneration(stdata<T> &StreamData, database<T> &CurrentData, runinfo<T> &RunData, database<T> &Leaders)
{
    // current generation
    int const currentGeneration = RunData.currentGeneration;

    // Next generation
    int const newGeneration = currentGeneration + 1;

    // Probabilty at each generation
    T *generationProbabilty = RunData.generationProbabilty.data();

    // Get the pointer to the function values
    T *Fvalue = CurrentData.fValue.data();

    // Total number of function values
    int const nCurrentSamplePoints = CurrentData.size();

    {
        T optimumP(T{});
        T optimumCoefVar(T{});

        // Probability of the current generation
        T const PJ = generationProbabilty[currentGeneration];

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
        generationProbabilty[newGeneration] = T{1};
        StreamData.eachPopulationSize[newGeneration] = StreamData.lastPopulationSize;
    }

    std::vector<T> weight(nCurrentSamplePoints);

    {
        // Get the PJ1 - PJ
        T const probabiltyDiff = generationProbabilty[newGeneration] - generationProbabilty[currentGeneration];

        // The function value is in log space
        for (int i = 0; i < nCurrentSamplePoints; i++)
        {
            weight[i] = Fvalue[i] * probabiltyDiff;
        }
    }

    stats s;
    T const weightMax = s.maxelement<T, T>(weight);

    std::for_each(weight.begin(), weight.end(), [&](T &w_i) { w_i = std::exp(w_i - weightMax); });

    if (optParams.Display)
    {
        io f;
        f.printMatrix<T>("Weight matrix", weight.data(), 1, nCurrentSamplePoints);
    }

    // Compute the weight sum
    T const weightSum = s.sum<T, T>(weight);

    // Normalize the weight
    std::for_each(weight.begin(), weight.end(), [&](T &w_i) { w_i /= weightSum; });

    if (optParams.Display)
    {
        io f;
        f.printMatrix<T>("Normalized Weight matrix", weight.data(), 1, nCurrentSamplePoints);
    }

    {
        // Keep the value for computing evidence
        RunData.logselection[currentGeneration] = std::log(weightSum) + weightMax - std::log(static_cast<T>(nCurrentSamplePoints));

        if (optParams.Display)
        {
            io f;
            f.printMatrix<T>("log selection matrix", RunData.logselection.data(), 1, newGeneration);
        }
    }

    // Compute the current Generation CoefVar
    RunData.CoefVar[currentGeneration] = s.coefvar<T, T>(weight);

    if (optParams.Display)
    {
        io f;
        f.printMatrix<T>("CoefVar matrix", RunData.CoefVar.data(), 1, newGeneration);
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
        std::vector<T> thetaMean(nDimSamplePoints);

        for (int i = 0; i < nDimSamplePoints; i++)
        {
            arrayWrapper<T> Parray(CurrentData.samplePoints.data() + i, nSize, nDimSamplePoints);
            thetaMean[i] = std::inner_product(weight.begin(), weight.end(), Parray.begin(), T{});
        }

        std::copy(thetaMean.begin(), thetaMean.end(), RunData.meantheta.data() + currentGeneration * nDimSamplePoints);

        if (optParams.Display)
        {
            io f;
            f.printMatrix<int>("Mean of theta = [", thetaMean.data(), 1, nDimSamplePoints, "]\n----------------------------------------\n");
        }

        {
            T *Covariance = RunData.covariance.data();

            for (int i = 0; i < nDimSamplePoints; i++)
            {
                arrayWrapper<T> iArray(CurrentData.samplePoints.data() + i, nSize, nDimSamplePoints);

                T const iMean = thetaMean[i];

                for (int j = i; j < nDimSamplePoints; j++)
                {
                    arrayWrapper<T> jArray(CurrentData.samplePoints.data() + j, nSize, nDimSamplePoints);

                    T const jMean = thetaMean[j];

                    T S(0);
                    for (auto i = iArray.begin(), j = jArray.begin(), k = weight.begin(); i != iArray.end(); i++, j++, k++)
                    {
                        T const d1 = *i - iMean;
                        T const d2 = *j - jMean;
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
            if (!isSelfAdjointMatrixPositiveDefinite<T>(Covariance, nDimSamplePoints))
            {
                // We force it to be positive definite
                forceSelfAdjointMatrixPositiveDefinite<T>(Covariance, nDimSamplePoints);
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

template <typename T, typename TOut>
TOut CoefVar(T const *FunValues, int const nFunValues, int const Stride, T const PJ1, T const PJ, T const Tolerance)
{
    arrayWrapper<T> iArray(FunValues, nFunValues, Stride);

    std::vector<TOut> weight(iArray.size());

    stats s;

    {
        //Find the maximum value in the array of FunValues of size nFunValues
        T const fMaxValue = s.maxelement<T>(iArray);

        TOut const diff = static_cast<TOut>(PJ1 - PJ);

        //Compute the weight
        for (std::size_t i = 0; i < iArray.size(); i++)
        {
            weight[i] = std::exp((iArray[i] - fMaxValue) * diff);
        }
    }

    { //Compute the summation of weight
        TOut const weightsum = s.sum<TOut, TOut>(weight);

        //Normalize the weight
        std::for_each(weight.begin(), weight.end(), [&](TOut &w) { w /= weightsum; });
    }

    //Compute the mean
    TOut const weightmean = s.mean<TOut, TOut>(weight);

    //Compute the standard deviation
    TOut const weightstddev = s.stddev<TOut, TOut>(weight, weightmean);

    //return the square of the coefficient of variation (COV)
    return std::pow(weightstddev / weightmean - Tolerance, 2);
}

template <typename T, typename TOut>
TOut CoefVar(T const *FunValues, int const nFunValues, T const PJ1, T const PJ, T const Tolerance)
{
    std::vector<TOut> weight(nFunValues);

    stats s;

    {
        //Find the maximum value in the array of FunValues of size nFunValues
        T const fMaxValue = s.maxelement<T>(FunValues, nFunValues);

        TOut const diff = static_cast<TOut>(PJ1 - PJ);

        //Compute the weight
        for (int i = 0; i < nFunValues; i++)
        {
            weight[i] = std::exp((FunValues[i] - fMaxValue) * diff);
        }
    }

    { //Compute the summation of weight
        TOut const weightsum = s.sum<TOut, TOut>(weight);

        //Normalize the weight
        std::for_each(weight.begin(), weight.end(), [&](TOut &w) { w /= weightsum; });
    }

    //Compute the mean
    TOut const weightmean = s.mean<TOut, TOut>(weight);

    //Compute the standard deviation
    TOut const weightstddev = s.stddev<TOut, TOut>(weight, weightmean);

    //return the square of the coefficient of variation (COV)
    return std::pow(weightstddev / weightmean - Tolerance, 2);
}

template <typename T, typename TOut>
TOut CoefVar(std::vector<T> const &FunValues, T const PJ1, T const PJ, T const Tolerance)
{
    std::vector<TOut> weight(FunValues.size());

    stats s;

    {
        //Find the maximum value in the array of FunValues of size nFunValues
        T const fMaxValue = s.maxelement<T>(FunValues);

        TOut const diff = static_cast<TOut>(PJ1 - PJ);

        //Compute the weight
        for (std::size_t i = 0; i < FunValues.size(); i++)
        {
            weight[i] = std::exp((FunValues[i] - fMaxValue) * diff);
        }
    }

    { //Compute the summation of weight
        TOut const weightsum = s.sum<TOut, TOut>(weight);

        //Normalize the weight
        std::for_each(weight.begin(), weight.end(), [&](TOut &w) { w /= weightsum; });
    }

    //Compute the mean
    TOut const weightmean = s.mean<TOut, TOut>(weight);

    //Compute the standard deviation
    TOut const weightstddev = s.stddev<TOut, TOut>(weight, weightmean);

    //return the square of the coefficient of variation (COV)
    return std::pow(weightstddev / weightmean - Tolerance, 2);
}

template <typename T>
T CoefVarFun(T const *x)
{
    return CoefVar<T, T>(functionValues<T>, nFunctionValues, *x, pj<T>, coefVarPresetThreshold<T>);
}

} // namespace tmcmc
} // namespace umuq

#endif // UMUQ_TMCMCSTATS

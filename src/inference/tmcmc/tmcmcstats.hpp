#ifndef UMUQ_TMCMCSTATS_H
#define UMUQ_TMCMCSTATS_H

#include "datatype.hpp"
#include "numerics/function/fitfunction.hpp"
#include "numerics/eigenlib.hpp"
#include "numerics/random/psrandom.hpp"
#include "numerics/stats.hpp"
#include "numerics/multimin.hpp"
#include "inference/prior/priordistribution.hpp"
#include "io/io.hpp"
#include "misc/funcallcounter.hpp"

#include <cstddef>
#include <cmath>

#include <vector>
#include <memory>
#include <utility>
#include <limits>
#include <algorithm>
#include <numeric>

namespace umuq
{

namespace tmcmc
{

/*!
 * \ingroup TMCMC_Module
 *
 * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold.
 *
 * \tparam double Output data type (return output result (default is double))
 *
 * \param  FunValues   An array of log value
 * \param  nFunValues  Number of FunValues
 * \param  Stride      Element stride
 * \param  PJ1         \f$ p_{j+1} \f$
 * \param  PJ          \f$ p_j \f$
 * \param  Tolerance   A prescribed tolerance
 *
 * \returns double The square of the coefficient of variation (COV)
 */
double CoefVar(double const *FunValues, int const nFunValues, int const Stride, double const PJ1, double const PJ, double const Tolerance);

/*!
 * \ingroup TMCMC_Module
 *
 * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold.
 *
 * \param  FunValues   An array of log value
 * \param  nFunValues  Number of FunValues
 * \param  PJ1         \f$ p_{j+1} \f$
 * \param  PJ          \f$ p_j \f$
 * \param  Tolerance   A prescribed tolerance
 *
 * \returns double The square of the coefficient of variation (COV)
 */
double CoefVar(double const *FunValues, int const nFunValues, double const PJ1, double const PJ, double const Tolerance);

/*!
 * \ingroup TMCMC_Module
 *
 * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold.
 *
 * \param  FunValues   An array of log value
 * \param  PJ1         \f$ p_{j+1} \f$
 * \param  PJ          \f$ p_j \f$
 * \param  Tolerance   A prescribed tolerance
 *
 * \returns double The square of the coefficient of variation (COV)
 */
double CoefVar(std::vector<double> const &FunValues, double const PJ1, double const PJ, double const Tolerance);

/*!
 * \ingroup TMCMC_Module
 *
 * \brief Computes the square of the coefficient of variation (COV) of the plausibility weights to a prescribed threshold
 * This function is just a wrapper to call CoefVar function but has the proper Function type that can be used
 * in multidimensional minimization \sa umuq::functionMinimizer.
 *
 * \param x Input array of plausibility weights
 *
 * \returns The square of the coefficient of variation (COV) for the choice of input p
 */
double CoefVarFun(double const *x);

/*!
 * \ingroup TMCMC_Module
 *
 * \brief Pointer to array of function values
 *
 * \tparam double Data type
 */
static double *functionValues;

/*! Number of function values in functionValues array */
static int nFunctionValues;

/*! Current \f$ p_j \f$ */
static double pj;

/*! A preset threshold for coefficient of variation of the plausibility of weights. */
static double coefVarPresetThreshold;

} // namespace tmcmc

namespace tmcmc
{

/*! \class tmcmcStats
 * \ingroup TMCMC_Module
 *
 * \brief Statistic class for TMCMC algorithm
 */
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
    tmcmcStats(optimizationParameters const &OptParams, double const CoefVarPresetThreshold);

    /*!
     * \brief Move constructor, construct a new tmcmcStats object from an input object
     *
     * \param other  Input tmcmcStats object
     */
    tmcmcStats(tmcmcStats &&other);

    /*!
     * \brief Move assignment operator
     *
     * \param other
     * \return tmcmcStats&
     */
    tmcmcStats &operator=(tmcmcStats &&other);

    /*!
     * \brief Destroy the tmcmc Stats object
     *
     */
    ~tmcmcStats();

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
     * \return false If it encounters any problem
     */
    bool findOptimumP(double const *FunValues, int const nFunValues, double const PJ, double *OptimumP, double *OptimumCoefVar);

    /*!
     * \brief Find the optimum value of  \f$ p_j \f$ through direct search of the \f$ p \in [0, 4] \f$
     *
     * \param FunValues       Function values
     * \param nFunValues      Number of elements in FunValues
     * \param PJ              \f$ p_j \f$
     * \param OptimumP        Optimum value of p
     * \param OptimumCoefVar  Optimum Coef Var
     *
     * \return false If it encounters any problem
     */
    bool searchOptimumP(double const *FunValues, int const nFunValues, double const PJ, double *OptimumP, double *OptimumCoefVar);

    /*!
     * \brief This function, prepares and set the new generation from current sample points
     *
     * \param StreamData   IO data, which includes general information
     * \param CurrentData  Current generation of sample points
     * \param RunData      Running information data
     * \param Leaders      Leader generation of sample points
     *
     * \returns false If it encounters any problem
     */
    bool selectNewGeneration(stdata &StreamData, database &CurrentData, runinfo &RunData, database &Leaders);

protected:
    /*!
     * \brief Delete a tmcmcStats object copy construction
     *
     * Avoiding implicit generation of the copy constructor.
     */
    tmcmcStats(tmcmcStats const &) = delete;

    /*!
     * \brief Delete a tmcmcStats object assignment
     *
     * Avoiding implicit copy assignment.
     */
    tmcmcStats &operator=(tmcmcStats const &) = delete;

private:
    /*! Optimizer information */
    optimizationParameters optParams;

    /*! Create an instance of the function minimizer */
    std::unique_ptr<umuq::functionMinimizer<double>> fMinimizer;
};

tmcmcStats::tmcmcStats() : optParams(),
                           fMinimizer(new umuq::simplexNM2<double>)
{
    // First we have to set the minimizer dimension
    if (!fMinimizer->reset(1))
    {
        UMUQFAIL("Failed to set the minimizer dimension!");
    }

    // Set the preset threshold for coefficient of variation of the plausibility of weights to the default value of 1
    coefVarPresetThreshold = double{1};
}

tmcmcStats::tmcmcStats(optimizationParameters const &OptParams, double const CoefVarPresetThreshold) : optParams(OptParams)
{
    // Get the correct instance of the minimizer
    switch (static_cast<umuq::multimin::FunctionMinimizerTypes>(optParams.FunctionMinimizerType))
    {
    case (umuq::multimin::FunctionMinimizerTypes::SIMPLEXNM):
        fMinimizer.reset(new umuq::multimin::simplexNM<double>);
        break;
    case (umuq::multimin::FunctionMinimizerTypes::SIMPLEXNM2):
        fMinimizer.reset(new umuq::multimin::simplexNM2<double>);
        break;
    case (umuq::multimin::FunctionMinimizerTypes::SIMPLEXNM2RND):
        fMinimizer.reset(new umuq::multimin::simplexNM2Rnd<double>);
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
    coefVarPresetThreshold = CoefVarPresetThreshold;
}

tmcmcStats::tmcmcStats(tmcmcStats &&other) : optParams(std::move(other.optParams)),
                                             fMinimizer(std::move(other.fMinimizer)) {}

tmcmcStats &tmcmcStats::operator=(tmcmcStats &&other)
{
    optParams = std::move(other.optParams);
    fMinimizer = std::move(other.fMinimizer);
    return *this;
}

tmcmcStats::~tmcmcStats(){};

bool tmcmcStats::findOptimumP(double const *FunValues, int const nFunValues, double const PJ, double *OptimumP, double *OptimumCoefVar)
{
    // Set the global helper pointer & variables
    functionValues = const_cast<double *>(FunValues);
    nFunctionValues = nFunValues;
    pj = PJ;

    // Second, we have to set the function, input vector and stepsize in the function minimizer
    if (!fMinimizer->set(CoefVarFun, &pj, &optParams.Step))
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
        double *x = fMinimizer->getX();
        UMUQMSG("x =", x[0]);
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
            double *x = fMinimizer->getX();
            UMUQMSG(iter, ": ");
            UMUQMSG("CoefVar(p=", x[0], ") =", fMinimizer->getMin(), ", & characteristic size =", fMinimizer->size());
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
            UMUQMSG(fMinimizer->getName(), ", on CoefVarFun : ", iter, " iters, CoefVar(p)=", *OptimumCoefVar);
            UMUQMSG(((status == 0) ? "Converged to minimum at p = " : "Stopped at p = "));
            UMUQMSG(*OptimumP);
        }
        return (status ? (std::abs(fMinimizer->getMin()) <= optParams.Tolerance) : true);
    }
    UMUQFAILRETURN("Could not find the optimum value of p through optimization!");
}

bool tmcmcStats::searchOptimumP(double const *FunValues, int const nFunValues, double const PJ, double *OptimumP, double *OptimumCoefVar)
{
    double MinValp = double{};
    double MaxValp = double{4};

    // Initialize the step size to the prescribed user value
    double stepSize(optParams.Step);

    for (;;)
    {
        if (optParams.Display)
        {
            UMUQMSG("Search for optimum p in [", MinValp, ", ", MaxValp, "] with step size =", stepSize);
        }

        // Define some variables to use in the search
        bool found(false);

        // Maximum number of iterations
        std::size_t const MaxIter = static_cast<std::size_t>((MaxValp - MinValp) / stepSize);

        // Optimum value of p
        double optimumP(MinValp);

        // Max value of CoefVar
        double MaxCoefVar(std::numeric_limits<double>::max());

        // We search the space for the optimum value
        for (std::size_t iter = 0; iter < MaxIter; iter++)
        {
            double const p = MinValp + iter * stepSize;
            double const NewCoefVar = CoefVarFun(&p);

            if (NewCoefVar < MaxCoefVar)
            {
                MaxCoefVar = NewCoefVar;
                optimumP = p;

                if (MaxCoefVar <= coefVarPresetThreshold)
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
                UMUQMSG("Search on CoefVarFun : CoefVar(p)=", *OptimumCoefVar);
                UMUQMSG("Converged to minimum at p = ", *OptimumP);
            }
            return true;
        }
        else
        {
            // If optimumP is not within Tolerance, we can go back and retry with better refinement (more iterations)

            MinValp = optimumP - 10 * stepSize;
            if (MinValp < 0)
            {
                MinValp = double{};
            }

            MaxValp = optimumP + 10 * stepSize;
            if (MaxValp > double{4})
            {
                MaxValp = double{4};
            }

            stepSize /= double{10};
            if (stepSize < machinePrecision<double>)
            {
                // Fail:-1
                UMUQFAILRETURN("Could not find the optimum value of p through search!");
            }
        }
    }
    UMUQFAILRETURN("Could not find the optimum value of p through search!");
}

bool tmcmcStats::selectNewGeneration(stdata &StreamData, database &CurrentData, runinfo &RunData, database &Leaders)
{
    // current generation
    int const currentGeneration = RunData.currentGeneration;

    // Next generation
    int const newGeneration = currentGeneration + 1;

    // Probabilty at each generation
    double *generationProbabilty = RunData.generationProbabilty.data();

    // Get the pointer to the function values
    double *Fvalue = CurrentData.fValue.data();

    // Total number of function values
    int const nCurrentSamplePoints = static_cast<int>(CurrentData.size());

    {
        double optimumP(double{});
        double optimumCoefVar(double{});

        // Probability of the current generation
        double const PJ = generationProbabilty[currentGeneration];

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
        generationProbabilty[newGeneration] = double{1};
        StreamData.eachPopulationSize[newGeneration] = StreamData.lastPopulationSize;
    }

    std::vector<double> weight(nCurrentSamplePoints);

    {
        // Get the PJ1 - PJ
        double const probabiltyDiff = generationProbabilty[newGeneration] - generationProbabilty[currentGeneration];

        // The function value is in log space
        for (int i = 0; i < nCurrentSamplePoints; i++)
        {
            weight[i] = Fvalue[i] * probabiltyDiff;
        }
    }

    stats s;
    double const weightMax = s.maxelement(weight);

    std::for_each(weight.begin(), weight.end(), [&](double &w_i) { w_i = std::exp(w_i - weightMax); });

    if (optParams.Display)
    {
        io f;
        f.printMatrix("Weight matrix", weight.data(), 1, nCurrentSamplePoints);
    }

    // Compute the weight sum
    double const weightSum = s.sum(weight);

    // Normalize the weight
    std::for_each(weight.begin(), weight.end(), [&](double &w_i) { w_i /= weightSum; });

    if (optParams.Display)
    {
        io f;
        f.printMatrix("Normalized Weight matrix", weight.data(), 1, nCurrentSamplePoints);
    }

    {
        // Keep the value for computing evidence
        RunData.logselection[currentGeneration] = std::log(weightSum) + weightMax - std::log(static_cast<double>(nCurrentSamplePoints));

        if (optParams.Display)
        {
            io f;
            f.printMatrix("log selection matrix", RunData.logselection.data(), 1, newGeneration);
        }
    }

    // Compute the current Generation CoefVar
    RunData.CoefVar[currentGeneration] = s.coefvar(weight);

    if (optParams.Display)
    {
        io f;
        f.printMatrix("CoefVar matrix", RunData.CoefVar.data(), 1, newGeneration);
    }

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

        // Get the number of selection based on the probability
        umuq::randomdist::multinomialDistribution<double> multinomial(weight.data(), nCurrentSamplePoints, nSamples);

        multinomial.dist(Leaders.nSelection);

        if (optParams.Display)
        {
            io f;
            f.printMatrix("Number of selection = [", Leaders.nSelection, 1, nCurrentSamplePoints, "]\n----------------------------------------\n");
        }
    }

    // Total size of the sampling points array
    int const nSize = nCurrentSamplePoints * nDimSamplePoints;

    // Compute vectors of mean and standard deviation for each dimension
    std::vector<double> thetaMean(nDimSamplePoints);

    for (int i = 0; i < nDimSamplePoints; i++)
    {
        arrayWrapper<double> Parray(CurrentData.samplePoints.data() + i, nSize, nDimSamplePoints);
        thetaMean[i] = std::inner_product(weight.begin(), weight.end(), Parray.begin(), double{});
    }

    std::copy(thetaMean.begin(), thetaMean.end(), RunData.meantheta.data() + currentGeneration * nDimSamplePoints);

    if (optParams.Display)
    {
        io f;
        f.printMatrix("Mean of theta = [", thetaMean.data(), 1, nDimSamplePoints, "]\n----------------------------------------\n");
    }

    {
        double *Covariance = RunData.covariance.data();

        for (int i = 0; i < nDimSamplePoints; i++)
        {
            arrayWrapper<double> iArray(CurrentData.samplePoints.data() + i, nSize, nDimSamplePoints);

            double const iMean = thetaMean[i];

            for (int j = i; j < nDimSamplePoints; j++)
            {
                arrayWrapper<double> jArray(CurrentData.samplePoints.data() + j, nSize, nDimSamplePoints);

                double const jMean = thetaMean[j];

                double S(0);
                auto k = weight.begin();
                for (auto i = iArray.begin(), j = jArray.begin(); i != iArray.end(); i++, j++, k++)
                {
                    double const d1 = *i - iMean;
                    double const d2 = *j - jMean;
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
        if (!umuq::linearalgebra::isSelfAdjointMatrixPositiveDefinite(Covariance, nDimSamplePoints))
        {
            // We force it to be positive definite
            umuq::linearalgebra::forceSelfAdjointMatrixPositiveDefinite(Covariance, nDimSamplePoints);
        }

        if (optParams.Display)
        {
            io f;
            f.printMatrix("Covariance = [", Covariance, nDimSamplePoints, nDimSamplePoints, "]\n----------------------------------------\n");
        }
    }
    return true;
}

double CoefVar(double const *FunValues, int const nFunValues, int const Stride, double const PJ1, double const PJ, double const Tolerance)
{
    arrayWrapper<double> iArray(FunValues, nFunValues, Stride);

    std::vector<double> weight(iArray.size());

    stats s;

    {
        //Find the maximum value in the array of FunValues of size nFunValues
        double const fMaxValue = s.maxelement(iArray);

        double const diff = PJ1 - PJ;

        //Compute the weight
        for (std::size_t i = 0; i < iArray.size(); i++)
        {
            weight[i] = std::exp((iArray[i] - fMaxValue) * diff);
        }
    }

    { //Compute the summation of weight
        double const weightsum = s.sum(weight);

        //Normalize the weight
        std::for_each(weight.begin(), weight.end(), [&](double &w) { w /= weightsum; });
    }

    //Compute the mean
    double const weightmean = s.mean(weight);

    //Compute the standard deviation
    double const weightstddev = s.stddev(weight, weightmean);

    //return the square of the coefficient of variation (COV)
    return std::pow(weightstddev / weightmean - Tolerance, 2);
}

double CoefVar(double const *FunValues, int const nFunValues, double const PJ1, double const PJ, double const Tolerance)
{
    std::vector<double> weight(nFunValues);

    stats s;

    {
        //Find the maximum value in the array of FunValues of size nFunValues
        double const fMaxValue = s.maxelement(FunValues, nFunValues);

        double const diff = PJ1 - PJ;

        //Compute the weight
        for (int i = 0; i < nFunValues; i++)
        {
            weight[i] = std::exp((FunValues[i] - fMaxValue) * diff);
        }
    }

    { //Compute the summation of weight
        double const weightsum = s.sum(weight);

        //Normalize the weight
        std::for_each(weight.begin(), weight.end(), [&](double &w) { w /= weightsum; });
    }

    //Compute the mean
    double const weightmean = s.mean(weight);

    //Compute the standard deviation
    double const weightstddev = s.stddev(weight, weightmean);

    //return the square of the coefficient of variation (COV)
    return std::pow(weightstddev / weightmean - Tolerance, 2);
}

double CoefVar(std::vector<double> const &FunValues, double const PJ1, double const PJ, double const Tolerance)
{
    std::vector<double> weight(FunValues.size());

    stats s;

    {
        //Find the maximum value in the array of FunValues of size nFunValues
        double const fMaxValue = s.maxelement(FunValues);

        double const diff = PJ1 - PJ;

        //Compute the weight
        for (std::size_t i = 0; i < FunValues.size(); i++)
        {
            weight[i] = std::exp((FunValues[i] - fMaxValue) * diff);
        }
    }

    { //Compute the summation of weight
        double const weightsum = s.sum(weight);

        //Normalize the weight
        std::for_each(weight.begin(), weight.end(), [&](double &w) { w /= weightsum; });
    }

    //Compute the mean
    double const weightmean = s.mean(weight);

    //Compute the standard deviation
    double const weightstddev = s.stddev(weight, weightmean);

    //return the square of the coefficient of variation (COV)
    return std::pow(weightstddev / weightmean - Tolerance, 2);
}

double CoefVarFun(double const *x)
{
    return CoefVar(functionValues, nFunctionValues, *x, pj, coefVarPresetThreshold);
}

} // namespace tmcmc
} // namespace umuq

#endif // UMUQ_TMCMCSTATS

#ifndef UMUQ_PRIORDISTRIBUTION_H
#define UMUQ_PRIORDISTRIBUTION_H

#include "core/core.hpp"
#include "datatype/priortype.hpp"
#include "numerics/density.hpp"
#include "numerics/random/psrandom.hpp"

namespace umuq
{

/*! \class priorDistribution
 * \ingroup Inference_Module
 * 
 * \brief Prior distribution which is one of the:
 * 
 * <table>
 * <caption id="multi_row">Prior distribution types</caption>
 * <tr><th> Index number <th> Prior distribution type        
 * <tr><td> 0       <td> UNIFORM  
 * <tr><td> 1       <td> GAUSSIAN    
 * <tr><td> 2       <td> EXPONENTIAL     
 * <tr><td> 3       <td> GAMMA 
 * <tr><td> 4       <td> COMPOSITE
 * </table>
 * 
 * \tparam RealType Data type 
 * 
 * USE: <br>
 * To use the priorDistribution object:
 * - First, construct a new prior Distribution object with problem dimension and the prior type. <br>
 *   \code
 *      int const  problemDimension = 4;
 *      priorTypes problemPriorTypes = priorTypes::UNIFORM;
 * 
 *      priorDistribution<double> prior(problemDimension, problemPriorTypes);
 *   \endcode <br>
 *   In case, the prior type is not known yet, you should reset the priorDistribution later in the code with the 
 *   correct problem dimension and the corresponding prior type. <br>
 *   \code
 *      priorDistribution<double> prior();
 *      \\ ...
 *      int const  problemDimension = 4;
 *      priorTypes problemPriorTypes = priorTypes::UNIFORM;
 *      
 *      prior.reset(problemDimension, problemPriorTypes);
 *   \endcode <br>
 *   \sa umuq::priorTypes. <br>
 *   \sa reset. <br>
 * 
 * - Second, set the priorDistribution parameters. <br> 
 *   \sa set.
 * 
 * - Third, call the member function. <br>
 *   You can call the probability density function (pdf). <br>
 *    or <br>
 *   logarithm probability density function (logpdf). <br>
 *   \sa pdf. <br>
 *   \sa logpdf. <br>
 *   You can also sample from the distribution using sample member function (sample). <br>
 *   \sa sample. <br>
 * 
 * - Forth, call any other member function.
 * 
 * \note
 * - For multi threaded and multi processors application the Random Number Generator object must be initialized before.
 * \sa umuq::psrandom
 * \sa umuq::psrandom::init
 * \sa umuq::psrandom::setState
 * 
 */
template <typename RealType>
class priorDistribution
{
  public:
    /*!
     * \brief Construct a new prior Distribution object
     * 
     */
    priorDistribution();

    /*!
     * \brief Construct a new prior Distribution object
     * 
     * \param probdim  Problem dimension
     * \param prior    Prior type (0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite)
     */
    priorDistribution(int const probdim, umuq::priorTypes const prior = umuq::priorTypes::UNIFORM);

    /*!
     * \brief Construct a new prior Distribution object
     * 
     * \param probdim  Problem dimension
     * \param prior    Prior type (0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite)
     */
    priorDistribution(int const probdim, int const prior);

    /*!
     * \brief Move constructor, construct a new priorDistribution object from input priorDistribution object
     * 
     * \param other priorDistribution object
     */
    priorDistribution(priorDistribution<RealType> &&other);

    /*!
     * \brief Move assignment operator
     * 
     * \param other priorDistribution object
     * 
     * \returns priorDistribution<RealType>& 
     */
    priorDistribution<RealType> &operator=(priorDistribution<RealType> &&other);

    /*!
     * \brief Destroy the prior Distribution object
     * 
     */
    ~priorDistribution();

    /*!
     * \brief Reset the priorDistribution object size & type
     * 
     * \param probdim  Problem dimension
     * \param prior    Prior type (0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite)
     * 
     * \returns false  If there is not enough memory or wrong prior type
     */
    bool reset(int const probdim, umuq::priorTypes const prior = umuq::priorTypes::UNIFORM);

    /*!
     * \brief Reset the priorDistribution object size & type
     * 
     * \param probdim  Problem dimension
     * \param prior    Prior type (0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite)
     * 
     * \returns false  If there is not enough memory or wrong prior type
     */
    bool reset(int const probdim, int const prior);

    /*!
     * \brief Set the priorDistribution parameters
     * 
     * \param Param1          First parameter for a prior distribution  
     * \param Param2          Second parameter for a prior distribution  
     * \param compositeprior  Composite priors type
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool set(RealType const *Param1, RealType const *Param2, umuq::priorTypes const *compositeprior = nullptr);

    /*!
     * \brief Set the priorDistribution parameters
     * 
     * \param Param1          First parameter for a prior distribution  
     * \param Param2          Second parameter for a prior distribution  
     * \param compositeprior  Composite priors type
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool set(RealType const *Param1, RealType const *Param2, int const *compositeprior);

    /*!
     * \brief Set the priorDistribution parameters
     * 
     * \param Param1          First parameter for a prior distribution  
     * \param Param2          Second parameter for a prior distribution  
     * \param compositeprior  Composite priors type
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool set(std::vector<RealType> const &Param1, std::vector<RealType> const &Param2, std::vector<umuq::priorTypes> const &compositeprior = EmptyVector<umuq::priorTypes>);

    /*!
     * \brief Set the priorDistribution parameters
     * 
     * \param Param1          First parameter for a prior distribution  
     * \param Param2          Second parameter for a prior distribution  
     * \param compositeprior  Composite priors type
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool set(std::vector<RealType> const &Param1, std::vector<RealType> const &Param2, std::vector<int> const &compositeprior);

    /*!
     * \brief Get the dimension
     * 
     * \returns int Dimension of the problem
     */
    inline int getDim();

    /*!
     * \brief Get the prior type
     * 
     * \returns umuq::priorTypes prior type
     */
    inline umuq::priorTypes getpriorType();

    /*!
     * \brief Get the Prior Types for the composite prior
     * 
     * \returns umuq::priorTypes* Prior Types
     */
    inline umuq::priorTypes *getPriorTypes();

    /*!
     * \brief Probability density function (pdf)
     * 
     * \param x  Input point
     *  
     * \returns RealType Returns the probability density function (pdf) evaluated in x
     */
    RealType pdf(RealType const *x);

    /*!
     * \brief Probability density function (pdf)
     * 
     * \param x  Input point
     *  
     * \returns RealType Returns the probability density function (pdf) evaluated in x
     */
    RealType pdf(std::vector<RealType> const &x);

    /*!
     * \brief Logarithm of the probability density function
     * 
     * \param x  Input point
     * 
     * \returns RealType Returns the logarithm probability density function (pdf) evaluated in x
     */
    RealType logpdf(RealType const *x);

    /*!
     * \brief Logarithm of the probability density function
     * 
     * \param x  Input point
     * 
     * \returns RealType Returns the logarithm probability density function (pdf) evaluated in x
     */
    RealType logpdf(std::vector<RealType> const &x);

    /*!
     * \brief Create samples based on the prior distribution type
     * 
     * \param x Samples 
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool sample(RealType *x);

    /*!
     * \brief Create samples based on the prior distribution type
     * 
     * \param x Samples 
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool sample(std::vector<RealType> &x);

  protected:
    /*!
     * \brief Delete a priorDistribution object copy construction
     * 
     * Avoiding implicit generation of the copy constructor.
     */
    priorDistribution(priorDistribution<RealType> const &) = delete;

    /*!
     * \brief Delete a priorDistribution object assignment
     * 
     * Avoiding implicit copy assignment.
     * 
     * \returns priorDistribution<RealType>& 
     */
    priorDistribution<RealType> &operator=(priorDistribution<RealType> const &) = delete;

  private:
    /*! Problem Dimension */
    int nDim;

    /*!
     * Prior type which is one of : <br>
     * 0: uniform, 1: gaussian, 2: exponential, 3: gamma, 4:composite<br>
     * 
     * \sa umuq::priorTypes
     */
    umuq::priorTypes priorType;

    /*! Composite distribution prior */
    std::unique_ptr<umuq::priorTypes[]> compositePrior;

  private:
    /*! Flat (Uniform) distribution */
    std::unique_ptr<umuq::density::uniformDistribution<RealType>> uniformDist;

    /*! The Multivariate Gaussian Distribution */
    std::unique_ptr<umuq::density::multivariateGaussianDistribution<RealType>> multivariateGaussianDist;

    /*! The exponential distribution */
    std::unique_ptr<umuq::density::exponentialDistribution<RealType>> exponentialDist;

    /*! The Gamma distribution */
    std::unique_ptr<umuq::density::gammaDistribution<RealType>> gammaDist;

    /*! The Gaussian distribution */
    std::unique_ptr<umuq::density::gaussianDistribution<RealType>> gaussianDist;

  private:
    /*!
     * The below data are only used for composite prior distribution
     * indexing of the points
     */
    std::vector<int> uniformIndex;
    std::vector<int> gaussianIndex;
    std::vector<int> exponentialIndex;
    std::vector<int> gammaIndex;

    /*! Input points */
    std::vector<RealType> uniformPoints;
    std::vector<RealType> gaussianPoints;
    std::vector<RealType> exponentialPoints;
    std::vector<RealType> gammaPoints;
};

template <typename RealType>
priorDistribution<RealType>::priorDistribution() : nDim(0),
                                                   priorType(umuq::priorTypes::UNIFORM),
                                                   compositePrior(nullptr),
                                                   uniformDist(nullptr),
                                                   multivariateGaussianDist(nullptr),
                                                   exponentialDist(nullptr),
                                                   gammaDist(nullptr),
                                                   gaussianDist(nullptr) {}

template <typename RealType>
priorDistribution<RealType>::priorDistribution(int const probdim, umuq::priorTypes const prior) : nDim(probdim),
                                                                                                  priorType(prior),
                                                                                                  compositePrior(nullptr),
                                                                                                  uniformDist(nullptr),
                                                                                                  multivariateGaussianDist(nullptr),
                                                                                                  exponentialDist(nullptr),
                                                                                                  gammaDist(nullptr),
                                                                                                  gaussianDist(nullptr)
{
    switch (priorType)
    {
    case umuq::priorTypes::UNIFORM:
        break;
    case umuq::priorTypes::GAUSSIAN:
        break;
    case umuq::priorTypes::EXPONENTIAL:
        break;
    case umuq::priorTypes::GAMMA:
        break;
    case umuq::priorTypes::COMPOSITE:
        break;
    default:
        UMUQFAIL("Unknown prior distribution type!");
        break;
    };
}

template <typename RealType>
priorDistribution<RealType>::priorDistribution(int const probdim, int const prior) : priorDistribution(probdim, static_cast<umuq::priorTypes>(prior)) {}

template <typename RealType>
priorDistribution<RealType>::priorDistribution(priorDistribution<RealType> &&other) : nDim(other.nDim),
                                                                                      priorType(other.priorType),
                                                                                      compositePrior(std::move(other.compositePrior)),
                                                                                      uniformDist(std::move(other.uniformDist)),
                                                                                      multivariateGaussianDist(std::move(other.multivariateGaussianDist)),
                                                                                      exponentialDist(std::move(other.exponentialDist)),
                                                                                      gammaDist(std::move(other.gammaDist)),
                                                                                      gaussianDist(std::move(other.gaussianDist)),
                                                                                      uniformIndex(std::move(other.uniformIndex)),
                                                                                      gaussianIndex(std::move(other.gaussianIndex)),
                                                                                      exponentialIndex(std::move(other.exponentialIndex)),
                                                                                      gammaIndex(std::move(other.gammaIndex)),
                                                                                      uniformPoints(std::move(other.uniformPoints)),
                                                                                      gaussianPoints(std::move(other.gaussianPoints)),
                                                                                      exponentialPoints(std::move(other.exponentialPoints)),
                                                                                      gammaPoints(std::move(other.gammaPoints))
{
    switch (priorType)
    {
    case umuq::priorTypes::UNIFORM:
        break;
    case umuq::priorTypes::GAUSSIAN:
        break;
    case umuq::priorTypes::EXPONENTIAL:
        break;
    case umuq::priorTypes::GAMMA:
        break;
    case umuq::priorTypes::COMPOSITE:
        break;
    default:
        UMUQFAIL("Unknown prior distribution type!");
        break;
    };
}

template <typename RealType>
priorDistribution<RealType> &priorDistribution<RealType>::operator=(priorDistribution<RealType> &&other)
{
    nDim = other.nDim;
    priorType = other.priorType;
    switch (priorType)
    {
    case umuq::priorTypes::UNIFORM:
        break;
    case umuq::priorTypes::GAUSSIAN:
        break;
    case umuq::priorTypes::EXPONENTIAL:
        break;
    case umuq::priorTypes::GAMMA:
        break;
    case umuq::priorTypes::COMPOSITE:
        break;
    default:
        UMUQFAIL("Unknown prior distribution type!");
        break;
    };
    compositePrior = std::move(other.compositePrior);
    uniformDist = std::move(other.uniformDist);
    multivariateGaussianDist = std::move(other.multivariateGaussianDist);
    exponentialDist = std::move(other.exponentialDist);
    gammaDist = std::move(other.gammaDist);
    gaussianDist = std::move(other.gaussianDist);
    uniformIndex = std::move(other.uniformIndex);
    gaussianIndex = std::move(other.gaussianIndex);
    exponentialIndex = std::move(other.exponentialIndex);
    gammaIndex = std::move(other.gammaIndex);
    uniformPoints = std::move(other.uniformPoints);
    gaussianPoints = std::move(other.gaussianPoints);
    exponentialPoints = std::move(other.exponentialPoints);
    gammaPoints = std::move(other.gammaPoints);

    return *this;
}

template <typename RealType>
priorDistribution<RealType>::~priorDistribution() {}

template <typename RealType>
bool priorDistribution<RealType>::reset(int const probdim, umuq::priorTypes const prior)
{
    nDim = probdim;
    priorType = prior;
    switch (priorType)
    {
    case umuq::priorTypes::UNIFORM:
        break;
    case umuq::priorTypes::GAUSSIAN:
        break;
    case umuq::priorTypes::EXPONENTIAL:
        break;
    case umuq::priorTypes::GAMMA:
        break;
    case umuq::priorTypes::COMPOSITE:
        break;
    default:
        UMUQFAILRETURN("Unknown prior distribution type!");
        break;
    };
    if (compositePrior)
    {
        compositePrior.reset(nullptr);
    }
    if (uniformDist)
    {
        uniformDist.reset(nullptr);
    }
    if (multivariateGaussianDist)
    {
        multivariateGaussianDist.reset(nullptr);
    }
    if (exponentialDist)
    {
        exponentialDist.reset(nullptr);
    }
    if (gammaDist)
    {
        gammaDist.reset(nullptr);
    }
    if (gaussianDist)
    {
        gaussianDist.reset(nullptr);
    }
    return true;
}

template <typename RealType>
bool priorDistribution<RealType>::reset(int const probdim, int const prior)
{
    return reset(probdim, static_cast<umuq::priorTypes>(prior));
}

template <typename RealType>
bool priorDistribution<RealType>::set(RealType const *Param1, RealType const *Param2, umuq::priorTypes const *compositeprior)
{
    switch (priorType)
    {
    case umuq::priorTypes::UNIFORM:
        try
        {
            uniformDist.reset(new umuq::density::uniformDistribution<RealType>(Param1, Param2, nDim * 2));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        break;
    case umuq::priorTypes::GAUSSIAN:
        try
        {
            multivariateGaussianDist.reset(new umuq::density::multivariateGaussianDistribution<RealType>(Param1, Param2, nDim * 2));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        break;
    case umuq::priorTypes::EXPONENTIAL:
        try
        {
            exponentialDist.reset(new umuq::density::exponentialDistribution<RealType>(Param1, nDim * 2));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        break;
    case umuq::priorTypes::GAMMA:
        try
        {
            gammaDist.reset(new umuq::density::gammaDistribution<RealType>(Param1, Param2, nDim * 2));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
        break;
    case umuq::priorTypes::COMPOSITE:
    {
        if (compositeprior)
        {
            try
            {
                compositePrior.reset(new umuq::priorTypes[nDim]());
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }

            std::copy(compositeprior, compositeprior + nDim, compositePrior.get());
        }
        else
        {
            UMUQFAILRETURN("Failed to provide composite prior types for each dimension!");
        }

        if (multivariateGaussianDist)
        {
            multivariateGaussianDist.reset(nullptr);
        }

        int nUNIFORM(0);
        int nGAUSSIAN(0);
        int nEXPONENTIAL(0);
        int nGAMMA(0);

        for (int i = 0; i < nDim; i++)
        {
            switch (compositePrior[i])
            {
            case umuq::priorTypes::UNIFORM:
                nUNIFORM++;
                break;
            case umuq::priorTypes::GAUSSIAN:
                nGAUSSIAN++;
                break;
            case umuq::priorTypes::EXPONENTIAL:
                nEXPONENTIAL++;
                break;
            case umuq::priorTypes::GAMMA:
                nGAMMA++;
                break;
            default:
                UMUQFAILRETURN("Unknown prior distribution type!");
                break;
            };
        }

        if (nUNIFORM)
        {
            uniformIndex.resize(nUNIFORM);
            uniformPoints.resize(nUNIFORM);
        }
        if (nGAUSSIAN)
        {
            gaussianIndex.resize(nGAUSSIAN);
            gaussianPoints.resize(nGAUSSIAN);
        }
        if (nEXPONENTIAL)
        {
            exponentialIndex.resize(nEXPONENTIAL);
            exponentialPoints.resize(nEXPONENTIAL);
        }
        if (nGAMMA)
        {
            gammaIndex.resize(nGAMMA);
            gammaPoints.resize(nGAMMA);
        }

        std::vector<RealType> uparam1(nUNIFORM);
        std::vector<RealType> uparam2(nUNIFORM);
        std::vector<RealType> nparam1(nGAUSSIAN);
        std::vector<RealType> nparam2(nGAUSSIAN);
        std::vector<RealType> eparam1(nEXPONENTIAL);
        std::vector<RealType> gparam1(nGAMMA);
        std::vector<RealType> gparam2(nGAMMA);

        nUNIFORM = 0;
        nGAUSSIAN = 0;
        nEXPONENTIAL = 0;
        nGAMMA = 0;

        for (int i = 0; i < nDim; i++)
        {
            switch (compositePrior[i])
            {
            case umuq::priorTypes::UNIFORM:
                uparam1[nUNIFORM] = Param1[i];
                uparam2[nUNIFORM] = Param2[i];
                uniformIndex[nUNIFORM] = i;
                nUNIFORM++;
                break;
            case umuq::priorTypes::GAUSSIAN:
                nparam1[nGAUSSIAN] = Param1[i];
                nparam2[nGAUSSIAN] = Param2[i];
                gaussianIndex[nGAUSSIAN] = i;
                nGAUSSIAN++;
                break;
            case umuq::priorTypes::EXPONENTIAL:
                eparam1[nEXPONENTIAL] = Param1[i];
                exponentialIndex[nEXPONENTIAL] = i;
                nEXPONENTIAL++;
                break;
            case umuq::priorTypes::GAMMA:
                gparam1[nGAMMA] = Param1[i];
                gparam2[nGAMMA] = Param2[i];
                gammaIndex[nGAMMA] = i;
                nGAMMA++;
                break;
            default:
                UMUQFAILRETURN("Unknown prior distribution type!");
                break;
            };
        }

        if (nUNIFORM)
        {
            try
            {
                uniformDist.reset(new umuq::density::uniformDistribution<RealType>(uparam1.data(), uparam2.data(), nUNIFORM * 2));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
        else
        {
            if (uniformDist)
            {
                uniformDist.reset(nullptr);
            }
        }
        if (nGAUSSIAN)
        {
            try
            {
                gaussianDist.reset(new umuq::density::gaussianDistribution<RealType>(nparam1.data(), nparam2.data(), nGAUSSIAN * 2));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
        else
        {
            gaussianDist.reset(nullptr);
        }
        if (nEXPONENTIAL)
        {
            try
            {
                exponentialDist.reset(new umuq::density::exponentialDistribution<RealType>(eparam1.data(), nEXPONENTIAL));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
        else
        {
            if (exponentialDist)
            {
                exponentialDist.reset(nullptr);
            }
        }
        if (nGAMMA)
        {
            try
            {
                gammaDist.reset(new umuq::density::gammaDistribution<RealType>(gparam1.data(), gparam2.data(), nGAMMA * 2));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
        else
        {
            if (gammaDist)
            {
                gammaDist.reset(nullptr);
            }
        }
    }

    break;
    default:
        UMUQFAILRETURN("Unknown prior distribution type!");
        break;
    };

    return true;
}

template <typename RealType>
bool priorDistribution<RealType>::set(RealType const *Param1, RealType const *Param2, int const *compositeprior)
{
    if (compositeprior)
    {
        std::vector<umuq::priorTypes> CompositePrior(nDim);
        std::transform(compositeprior, compositeprior + nDim, CompositePrior.begin(), [](int const c) -> umuq::priorTypes { return static_cast<umuq::priorTypes>(c); });
        return set(Param1, Param2, CompositePrior.data());
    }
    return set(Param1, Param2);
}

template <typename RealType>
bool priorDistribution<RealType>::set(std::vector<RealType> const &Param1, std::vector<RealType> const &Param2, std::vector<umuq::priorTypes> const &compositeprior)
{
    return compositeprior.size() ? set(Param1.data(), Param2.data(), compositeprior.data()) : set(Param1.data(), Param2.data());
}

template <typename RealType>
bool priorDistribution<RealType>::set(std::vector<RealType> const &Param1, std::vector<RealType> const &Param2, std::vector<int> const &compositeprior)
{
    return compositeprior.size() ? set(Param1.data(), Param2.data(), compositeprior.data()) : set(Param1.data(), Param2.data());
}

template <typename RealType>
inline int priorDistribution<RealType>::getDim()
{
    return nDim;
}

template <typename RealType>
inline umuq::priorTypes priorDistribution<RealType>::getpriorType()
{
    return priorType;
}

template <typename RealType>
inline umuq::priorTypes *priorDistribution<RealType>::getPriorTypes()
{
    return compositePrior.get();
}

template <typename RealType>
RealType priorDistribution<RealType>::pdf(RealType const *x)
{
    switch (priorType)
    {
    case umuq::priorTypes::UNIFORM:
        return uniformDist->f(x);
        break;
    case umuq::priorTypes::GAUSSIAN:
        return multivariateGaussianDist->f(x);
        break;
    case umuq::priorTypes::EXPONENTIAL:
        return exponentialDist->f(x);
        break;
    case umuq::priorTypes::GAMMA:
        return gammaDist->f(x);
        break;
    case umuq::priorTypes::COMPOSITE:
        RealType sum(1);
        if (uniformDist)
        {
            int j(0);
            for (auto i : uniformIndex)
            {
                uniformPoints[j++] = x[i];
            }
            sum *= uniformDist->f(uniformPoints.data());
        }
        if (gaussianDist)
        {
            int j(0);
            for (auto i : gaussianIndex)
            {
                gaussianPoints[j++] = x[i];
            }
            sum *= gaussianDist->f(gaussianPoints.data());
        }
        if (exponentialDist)
        {
            int j(0);
            for (auto i : exponentialIndex)
            {
                exponentialPoints[j++] = x[i];
            }
            sum *= exponentialDist->f(exponentialPoints.data());
        }
        if (gammaDist)
        {
            int j(0);
            for (auto i : gammaIndex)
            {
                gammaPoints[j++] = x[i];
            }
            sum *= gammaDist->f(gammaPoints.data());
        }
        return sum;
        break;
    }
    UMUQFAIL("Unknown Prior type!");
}

template <typename RealType>
RealType priorDistribution<RealType>::pdf(std::vector<RealType> const &x)
{
    return pdf(x.data());
}

template <typename RealType>
RealType priorDistribution<RealType>::logpdf(RealType const *x)
{
    switch (priorType)
    {
    case umuq::priorTypes::UNIFORM:
        return uniformDist->lf(x);
        break;
    case umuq::priorTypes::GAUSSIAN:
        return multivariateGaussianDist->lf(x);
        break;
    case umuq::priorTypes::EXPONENTIAL:
        return exponentialDist->lf(x);
        break;
    case umuq::priorTypes::GAMMA:
        return gammaDist->lf(x);
        break;
    case umuq::priorTypes::COMPOSITE:
        RealType sum(0);
        if (uniformDist)
        {
            int j(0);
            for (auto i : uniformIndex)
            {
                uniformPoints[j++] = x[i];
            }
            sum += uniformDist->lf(uniformPoints.data());
        }
        if (gaussianDist)
        {
            int j(0);
            for (auto i : gaussianIndex)
            {
                gaussianPoints[j++] = x[i];
            }
            sum += gaussianDist->lf(gaussianPoints.data());
        }
        if (exponentialDist)
        {
            int j(0);
            for (auto i : exponentialIndex)
            {
                exponentialPoints[j++] = x[i];
            }
            sum += exponentialDist->lf(exponentialPoints.data());
        }
        if (gammaDist)
        {
            int j(0);
            for (auto i : gammaIndex)
            {
                gammaPoints[j++] = x[i];
            }
            sum += gammaDist->lf(gammaPoints.data());
        }
        return sum;
        break;
    }
    UMUQFAIL("Unknown Prior type!");
}

template <typename RealType>
RealType priorDistribution<RealType>::logpdf(std::vector<RealType> const &x)
{
    return logpdf(x.data());
}

template <typename RealType>
bool priorDistribution<RealType>::sample(RealType *x)
{
    switch (priorType)
    {
    case umuq::priorTypes::UNIFORM:
        uniformDist->sample(x);
        return true;
        break;
    case umuq::priorTypes::GAUSSIAN:
        multivariateGaussianDist->sample(x);
        return true;
        break;
    case umuq::priorTypes::EXPONENTIAL:
        exponentialDist->sample(x);
        return true;
        break;
    case umuq::priorTypes::GAMMA:
        gammaDist->sample(x);
        return true;
        break;
    case umuq::priorTypes::COMPOSITE:
        if (uniformDist)
        {
            uniformDist->sample(uniformPoints);
            {
                int j(0);
                for (auto i : uniformIndex)
                {
                    x[i] = uniformPoints[j++];
                }
            }
        }
        if (gaussianDist)
        {
            gaussianDist->sample(gaussianPoints);
            {
                int j(0);
                for (auto i : gaussianIndex)
                {
                    x[i] = gaussianPoints[j++];
                }
            }
        }
        if (exponentialDist)
        {
            exponentialDist->sample(exponentialPoints);
            {
                int j(0);
                for (auto i : exponentialIndex)
                {
                    x[i] = exponentialPoints[j++];
                }
            }
        }
        if (gammaDist)
        {
            gammaDist->sample(gammaPoints);
            {
                int j(0);
                for (auto i : gammaIndex)
                {
                    x[i] = gammaPoints[j++];
                }
            }
        }
        return true;
        break;
    }
    UMUQFAIL("Unknown Prior type!");
}

template <typename RealType>
bool priorDistribution<RealType>::sample(std::vector<RealType> &x)
{
    return sample(x.data());
}

} // namespace umuq

#endif // UMUQ_PRIORDISTRIBUTION

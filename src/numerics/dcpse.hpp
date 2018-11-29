#ifndef UMUQ_DCPSE_H
#define UMUQ_DCPSE_H

#include "polynomials.hpp"
#include "factorial.hpp"
#include "eigenlib.hpp"
#include "knearestneighbors.hpp"
#include "primitive.hpp"
#include "stats.hpp"

namespace umuq
{

/*! 
 * \defgroup Numerics_Module Numerics module
 * This is the numerics module of %UMUQ providing all necessary classes of numerical computation.
 */

/*! 
 * \defgroup DCPSE Discretization-corrected PSE Operators
 * \ingroup Numerics_Module 
 */

/*! \class dcpse
 * \ingroup DCPSE
 * 
 * \brief General class for (DC-PSE) \link DCPSE Discretization-Corrected PSE Operators \endlink
 * 
 * \tparam RealType        Floating-point data type
 * \tparam DistanceType    Distance type for finding k nearest neighbors. 
 *                         (Default is a specialized class - \b kNearestNeighbor<RealType> with L2 distance)<br>
 *                         \sa umuq::NeighborDistanceTypes
 *                         \sa umuq::kNearestNeighbor.<br>
 * \tparam PolynomialType  Polynomial type used in building the vandermonde & vandermonde-like matrix
 *                         (Default is - \b polynomial<RealType> with monomials)<br>
 *                         \sa umuq::polynomials::PolynomialTypes.<br>
 * 
 * 
 * It creates a discretized differential operator and interpolators.
 * 
 * \todo
 * Currently the class works only for one term and it should be extended to multi terms
 */
template <typename RealType, umuq::NeighborDistanceTypes DistanceType = umuq::NeighborDistanceTypes::L2, class PolynomialType = polynomial<RealType>>
class dcpse
{
  public:
    /*!
     * \brief Construct a new dcpse object
     * 
     * \param ndim    Dimensionality
     * \param nterms  Number of terms (currently only one term is implemented)
     */
    explicit dcpse(int ndim, int nterms = 1);

    /*!
     * \brief Move constructor, construct a new dcpse object
     * 
     * \param other dcpse object
     */
    explicit dcpse(dcpse<RealType, DistanceType, PolynomialType> &&other);

    /*!
     * \brief Move assignment operator
     * 
     * \param other dcpse object
     * 
     * \returns dcpse<RealType, DistanceType, PolynomialType>& dcpse object
     */
    dcpse<RealType, DistanceType, PolynomialType> &operator=(dcpse<RealType, DistanceType, PolynomialType> &&other);

    /*!
     * \brief Destroy the dcpse object
     * 
     */
    ~dcpse();

    /*!
     * \brief Computes generalized DC-PSE differential operators on set of input points
     *
     * This function uses one set of points as input data to compute the generalized DC-PSE 
     * differential operators.<br>
     * If the degree of the differential operator is zero \f$ | \beta | = 0 \f$, suggests one 
     * should use the interpolator function not this one. 
     * 
     * \param dataPoints   Input data points
     * \param nDataPoints  Number of data points
     * \param beta         In multi-dimensional notation \f$ \beta=\left(\beta_1, \cdots, \beta_d \right). \f$<br>
     *                     Notation for partial derivatives:<br>
     *                     \f$  D^\beta = \frac{\partial^{|\beta|}}{\partial x_1^{\beta_1} \partial x_2^{\beta_2}\cdots\partial x_d^{\beta_d}}. \f$
     * \param order        Order of accuracy (default is 2nd order accurate)
     * \param nENN         Number of extra nearest neighbors to aid in case of singularity of the Vandermonde matrix (default is 2)
     * \param ratio        The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     * 
     */
    bool computeWeights(RealType *dataPoints, int const nDataPoints,
                        int *beta, int order = 2, int nENN = 2, RealType ratio = static_cast<RealType>(1));

    /*!
     * \brief Computes generalized DC-PSE differential operators on the set of query points.
     * 
     * This function uses one set of points as input data to compute the generalized DC-PSE 
     * differential opearators on the set of query points.<br>
     * If the degree of the differential operator is zero \f$ | \beta | = 0 \f$, means one should
     * use the interpolator function not this one. 
     * 
     * \param dataPoints        A pointer to input data
     * \param nDataPoints       Number of data points
     * \param queryDataPoints   A pointer to query data
     * \param nQueryDataPoints  Number of query data points
     * \param beta              In multi-dimensional notation \f$ \beta=\left(\beta_1, \cdots, \beta_d \right). \f$<br>
     *                          Notation for partial derivatives:<br>
     *                          \f$ D^\beta = \frac{\partial^{|\beta|}} {\partial x_1^{\beta_1} \partial x_2^{\beta_2}\cdots\partial x_d^{\beta_d}}. \f$
     * \param order             Order of accuracy (default is 2nd order accurate)
     * \param nENN              Number of extra nearest neighbors to aid in case of singularity of the Vandermonde matrix (default is 2)
     * \param ratio             The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     */
    bool computeWeights(RealType *dataPoints, int const nDataPoints, RealType *queryDataPoints, int const nQueryDataPoints,
                        int *beta, int order = 2, int nENN = 2, RealType ratio = static_cast<RealType>(1));

    /*! 
     * \brief Computes generalized DC-PSE interpolator operators on the set of points.
     * 
     * This function uses one set of points as input data to compute the generalized DC-PSE 
     * interpolator operators on them.
     * 
     * \param dataPoints   A pointer to input data 
     * \param nDataPoints  Number of data points
     * \param order        Order of accuracy (default is 2nd order accurate)
     * \param nENN         Number of extra nearest neighbors to aid in case of singularity of the Vandermonde matrix (default is 2)
     * \param ratio        The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     */
    bool computeInterpolatorWeights(RealType *dataPoints, int const nDataPoints,
                                    int order = 2, int nENN = 2, RealType ratio = static_cast<RealType>(1));

    /*!
     * \brief Computes generalized DC-PSE interpolator operators on the set of query points.
     * 
     * This function uses one set of points as input data to compute the generalized DC-PSE 
     * interpolator operators on the set of query points.
     * 
     * \param dataPoints        A pointer to input data 
     * \param nDataPoints       Number of data points
     * \param queryDataPoints   A pointer to query data 
     * \param nQueryDataPoints  Number of query data points
     * \param order             Order of accuracy (default is 2nd order accurate)
     * \param nENN              Number of extra nearest neighbors to aid in case of singularity of the Vandermonde matrix (default is 2)
     * \param ratio             The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     */
    bool computeInterpolatorWeights(RealType *dataPoints, int const nDataPoints, RealType *queryDataPoints, int const nQueryDataPoints,
                                    int order = 2, int nENN = 2, RealType ratio = static_cast<RealType>(1));

    /*!
     * \brief Evaluate a discretized DC-PSE operator from function values of input data and put the results as the query data function values
     * 
     * This function uses function values of input data and the weights of the operator which have 
     * been previously computed to compute the query values and put the results as the query data 
     * function values. <br>
     * At first it checks the computed kernel size to be equal to the number of query points times the 
     * size of monomials which has been previously computed for the required degree of the DC-PSE operator.
     * 
     * \param dataFunctionValues   Input data function values
     * \param nDataPoints          Number of data points
     * \param queryFunctionValues  Query data function value
     * \param nQueryDataPoints     Number of query data points
     */
    bool compute(RealType *dataFunctionValues, int const nDataPoints, RealType *queryFunctionValues, int const nQueryDataPoints);

    /*!
     * \brief Evaluate a discretized DC-PSE interpolation operator from function values of input data and put the 
     * interpolation results as the query data values
     * 
     * This function uses function values of input data and the weights of the interpolation operator which have 
     * been previously computed to compute the query values and put the results as the query data 
     * function values. <br>
     * At first it checks the computed kernel size to be equal to the number of query points times the 
     * size of monomials which has been previously computed for the required degree of DC-PSE operator
     * or interpolator.
     * 
     * \param dataFunctionValues   Input data function values
     * \param nDataPoints          Number of data points
     * \param queryFunctionValues  Query data function values
     * \param nQueryDataPoints     Number of query data points
     */
    bool interpolate(RealType const *dataFunctionValues, int const nDataPoints, RealType *queryFunctionValues, int const nQueryDataPoints);

    /*!
     * \brief A pointer to neighborhood kernel at index
     * 
     * \param index Index of a point (from query data points) to get its neighborhood kernel
     * 
     * \returns A (pointer to a) row of the nearest neighbors kernel values.
     */
    inline RealType *neighborhoodKernel(int const index) const;

    /*!
     * \brief A pointer to kernel array of all query points
     * 
     * \returns A pointer to kernel array of all query points
     */
    inline RealType *neighborhoodKernel() const;

    /*!
     * \brief Size of the neighborhood kernel which equals to the monomial size 
     * 
     * \returns Size of the neighborhood kernel
     */
    inline int neighborhoodKernelSize() const;

    /*!
     * \brief Order of accuracy of DC-PSE kernel at index
     * 
     * \param index Index number in nTerms array
     * 
     * \returns Order of accuracy of DC-PSE kernel at index
     */
    inline int orderofAccuracy(int const index = 0) const;

    /*!
     * \brief Prints the DC-PSE information
     * 
     */
    inline void printInfo() const;

    /*!
     * \brief Component-wise average neighbor spacing at index
     * 
     * Component-wise average neighbor spacing is defined as:<br> 
     * \f$ h = \frac{1}{N} \sum_{p=1}^{N}\left(|x_{1}-x_{p1}| + \cdots |x_{d} -x_{pd}| \right), \f$
     * 
     * \param index Index of a point (from data points) to get its average neighbor spacing
     * 
     * \returns A component-wise average neighbor spacing at index 
     */
    inline RealType averageSpace(int const index) const;

    /*!
     * \brief A pointer to component-wise average neighbor spacing
     * 
     * \returns A pointer to component-wise average neighbor spacing
     */
    inline RealType *averageSpace() const;

  private:
    /*!
     * \brief Delete a dcpse object copy construction
     * 
     * Avoiding implicit generation of the copy constructor.
     */
    dcpse(dcpse<RealType, DistanceType, PolynomialType> const &) = delete;

    /*!
     * \brief Delete a dcpse object assignment
     * 
     * Avoiding implicit copy assignment.
     */
    dcpse<RealType, DistanceType, PolynomialType> &operator=(dcpse<RealType, DistanceType, PolynomialType> const &) = delete;

  private:
    //! Dimension of space
    int nDim;

    //! Number of terms
    int nTerms;

    //! The monomial size
    /*! \f$ \text{monomialSize} = \left(\begin{matrix} r + d -1 \\ d \end{matrix}\right) \f$ */
    int dcMonomialSize;

    //! Size of the kernel
    int dcKernelSize;

    //! Order of accuracy for each term
    std::vector<int> Order;

    //! Operator kernel
    std::vector<RealType> dcKernel;

    //! k-NearestNeighbor Object
    std::unique_ptr<kNearestNeighbor<RealType, DistanceType>> KNN;

    //! Component-wise average neighbor spacing \f$ h = \frac{1}{N} \sum_{p=1}^{N}\left(|x_{1}-x_{p1}| + \cdots |x_{d} -x_{pd}| \right), \f$
    std::vector<RealType> h_average;

    //! The sign is chosen positive for odd \f$ | \beta | \f$ and negative for even \f$ | \beta | \f$
    RealType rhscoeff;
};

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
dcpse<RealType, DistanceType, PolynomialType>::dcpse(int ndim, int nterms) : nDim(ndim),
                                                                             nTerms(nterms),
                                                                             dcMonomialSize(0),
                                                                             dcKernelSize(0),
                                                                             Order(nterms)
{
    if (!std::is_floating_point<RealType>::value)
    {
        UMUQFAIL("This data type is not supported in this class!");
    }
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
dcpse<RealType, DistanceType, PolynomialType>::dcpse(dcpse<RealType, DistanceType, PolynomialType> &&other)
{
    nDim = other.nDim;
    nTerms = other.nTerms;
    dcMonomialSize = other.dcMonomialSize;
    dcKernelSize = other.dcKernelSize;
    Order = std::move(other.Order);
    dcKernel = std::move(other.dcKernel);
    KNN = std::move(other.KNN);
    h_average = std::move(other.h_average);
    rhscoeff = other.rhscoeff;
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
dcpse<RealType, DistanceType, PolynomialType> &dcpse<RealType, DistanceType, PolynomialType>::operator=(dcpse<RealType, DistanceType, PolynomialType> &&other)
{
    nDim = other.nDim;
    nTerms = other.nTerms;
    dcMonomialSize = other.dcMonomialSize;
    dcKernelSize = other.dcKernelSize;
    Order = std::move(other.Order);
    dcKernel = std::move(other.dcKernel);
    KNN = std::move(other.KNN);
    h_average = std::move(other.h_average);
    rhscoeff = other.rhscoeff;

    return *this;
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
dcpse<RealType, DistanceType, PolynomialType>::~dcpse() {}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
bool dcpse<RealType, DistanceType, PolynomialType>::computeWeights(RealType *dataPoints, int const nDataPoints, int *beta, int order, int nENN, RealType ratio)
{
    if (nDataPoints < 1)
    {
        UMUQFAILRETURN("Number of input data points are negative!");
    }

    // Extra check on the order
    order = (order > 0) ? order : 2;
    {
        int *o = Order.data();
        std::fill(o, o + nTerms, order);
    }

    // Extra check on the number of extra nearest neighbors
    nENN = (nENN > 0) ? nENN : 0;

    // Extra check on the ratio
    ratio = (ratio > 0) ? ratio : static_cast<RealType>(1);

    // \f$ |\beta| = \beta_1 + \cdots + \beta_d \f$
    int Beta = std::accumulate(beta, beta + nDim, 0);
    if (Beta == 0)
    {
        UMUQWARNING("Zero order degree derivative gives an approximation! \n If this is an interpolation use the interpolation function!");
    }

    int alphamin = !(Beta & 1);

    // \f$ (-1)^{|\beta|} \f$
    rhscoeff = alphamin ? static_cast<RealType>(1) : -static_cast<RealType>(1);

    // Create an instance of polynomial object with polynomial degree of \f$ |\beta| + r -1 \f$
    PolynomialType poly(nDim, order + Beta - 1);

    /*
     * Get the monomials size
     * \f$ \text{monomialSize} = \left(\begin{matrix} |\beta| + r + d -1 \\ d \end{matrix}\right) - \alpha_{\min} \f$
     */
    dcMonomialSize = poly.monomialsize() - alphamin;

    if (nDataPoints * dcMonomialSize > dcKernelSize)
    {
        dcKernelSize = nDataPoints * dcMonomialSize;
        try
        {
            // Make sure of the correct kernel size
            dcKernel.resize(dcKernelSize);

            h_average.resize(nDataPoints);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        dcKernelSize = nDataPoints * dcMonomialSize;
    }

    if (KNN)
    {
        if (nDataPoints != KNN->numInputdata() || nDataPoints != KNN->numQuerydata())
        {
            try
            {
                /*
                 * Finding K nearest neighbors
                 * The number of points K in the neighborhood of each point
                 * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
                 */
                KNN.reset(new kNearestNeighbor<RealType, DistanceType>(nDataPoints, nDim, dcMonomialSize + nENN));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
    }
    else
    {
        try
        {
            /*
             * Finding K nearest neighbors
             * The number of points K in the neighborhood of each point
             * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
             */
            KNN.reset(new kNearestNeighbor<RealType, DistanceType>(nDataPoints, nDim, dcMonomialSize + nENN));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    // If KNN requires covariance
    if (KNN->needsCovariance())
    {
        stats s;
        RealType *Covariance = s.covariance<RealType, RealType>(dataPoints, nDataPoints * nDim, nDim);
        KNN->setCovariance(Covariance);
        delete[] Covariance;
    }

    // Construct a kd-tree index & do nearest neighbors search
    KNN->buildIndex(dataPoints);

    /*
     * Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
     * \f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^RealType({\mathbf x})={\mathbf b}  \f$
     */
    EVectorX<RealType> RHSB(dcMonomialSize);
    {
        // Get a pointer to the monomial basis
        int *alpha = poly.monomialBasis();

        for (int i = 0, id = alphamin * nDim; i < dcMonomialSize; i++)
        {
            int maxalpha = 0;
            for (int j = 0; j < nDim; j++, id++)
            {
                maxalpha = std::max(maxalpha, std::abs(alpha[id] - beta[j]));
            }
            if (maxalpha)
            {
                RHSB(i) = RealType{};
            }
            else
            {
                RealType fact = static_cast<RealType>(1);
                std::for_each(beta, beta + nDim, [&](int const b_j) { fact *= factorial<RealType>(b_j); });
                RHSB(i) = rhscoeff * fact;
            }
        }

        /*!
         * \todo
         * Check this again
         */
        /* 
         * When applicable, and for stability reasons, set the zeroth moment to 5
         */
        if (rhscoeff > RealType{})
        {
            std::ptrdiff_t const id = alphamin * nDim;
            if (std::accumulate(alpha + id, alpha + id + nDim, 0) == 0)
            {
                RHSB(0) = static_cast<RealType>(5);
            }
        }
    }

    // Total number of nearest neighbors for each point
    int nNN = KNN->numNearestNeighbors();

    /*
     * Creating a transpose of the Vandermonde matrix
     * with the size of monomials * monomials \f$  = l \times l \f$
     */
    EMatrixX<RealType> VandermondeMatrixTranspose(dcMonomialSize, dcMonomialSize);
    EMatrixX<RealType> VandermondeMatrixTransposeImage(dcMonomialSize, nNN);

    // Matrix of exponential window function
    EVectorX<RealType> ExponentialWindowMatrix(dcMonomialSize);
    EVectorX<RealType> ExponentialWindowMatrixImage(nNN);

    // Matrix A of a linear system for the kernel coefficients
    EMatrixX<RealType> AM(dcMonomialSize, dcMonomialSize);

    // Matrix B
    EMatrixX<RealType> BMT(dcMonomialSize, dcMonomialSize);
    EMatrixX<RealType> BMTimage(dcMonomialSize, nNN);

    // ${\mathbf a}^RealType({\mathbf x})$ is the column vector of coefficients which is the solution of linear system
    EVectorX<RealType> SV(dcMonomialSize);

    // Array for keeping the component-wise L1 distances
    std::vector<RealType> L1Dist(nNN * nDim);

    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
    RealType *column = new RealType[dcMonomialSize + alphamin];

    std::vector<int> IndexId(dcMonomialSize);

    // Number of points with singular Vandermonde matrix
    int nPointsWithSingularVandermondeMatrix(0);

    // Loop over all points
    for (int i = 0; i < nDataPoints; i++)
    {
        std::ptrdiff_t const IdM = i * dcMonomialSize;
        std::ptrdiff_t const IdI = i * nDim;

        // A pointer to nearest neighbors indices of point i
        int *NearestNeighbors = KNN->NearestNeighbors(i);

        // A pointer to nearest neighbors square distances from the point i
        RealType *nnDist = KNN->NearestNeighborsDistances(i);

        /* 
         * For each point \f$ {\mathbf x} \f$ we define:
         * 
         * \f$
         * \left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x}_p - {\mathbf x} \right\}, 
         * \f$
         * 
         * as the set of vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points 
         * \f${\mathbf x}_p\f$ in the support of \f${\mathbf x}\f$.
         * 
         */
        {
            // pointer to query data
            RealType *Idata = dataPoints + IdI;

            // \f$ $\left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x} - {\mathbf x}_p \right\} \f$
            for (int j = 0, n = 0; j < nNN; j++)
            {
                std::ptrdiff_t const IdJ = NearestNeighbors[j] * nDim;

                // pointer to dataPoints (neighbors of i)
                RealType *Jdata = dataPoints + IdJ;

                for (int d = 0; d < nDim; d++, n++)
                {
                    L1Dist[n] = Idata[d] - Jdata[d];
                }
            }
        }

        // Compute component-wise average neighbor spacing
        RealType h_avg(0);
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](RealType const l_i) { h_avg += std::abs(l_i); });

        // Component-wise average neighbor spacing \f$ h \f$
        h_avg /= static_cast<RealType>(nNN);

        h_average[i] = h_avg;

        // Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
        RealType const byEpsilon = ratio / h_avg;
        RealType const byEpsilonSq = byEpsilon * byEpsilon;
        RealType const byEpsilonSqHalf = 0.5 * byEpsilonSq;
        RealType const byEpsilonPowerBeta = std::pow(byEpsilon, Beta);

        // Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](RealType &l_i) { l_i *= byEpsilon; });

        // Loop through the neighbors
        for (int j = 0; j < dcMonomialSize; j++)
        {
            // Id in the L1 distance list
            std::ptrdiff_t const Id = j * nDim;

            // Evaluates a monomial at a point \f$ {\mathbf x} \f$
            poly.monomialValue(L1Dist.data() + Id, column);

            EVectorMapType<RealType> columnV(column + alphamin, dcMonomialSize);

            // Fill the Vandermonde matrix column by column
            VandermondeMatrixTranspose.block(0, j, dcMonomialSize, 1) << columnV;
        }

        for (int j = 0; j < dcMonomialSize; j++)
        {
            ExponentialWindowMatrix(j) = std::exp(-nnDist[j] * byEpsilonSqHalf);
        }

        int dcVandermondeMatrixRank;

        {
            // LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<RealType>> lu(VandermondeMatrixTranspose);

            dcVandermondeMatrixRank = lu.rank();

            if (dcVandermondeMatrixRank < dcMonomialSize && dcVandermondeMatrixRank >= dcMonomialSize - nENN)
            {
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    IndexId[j] = lu.permutationQ().indices()(j);
                }
            }
        }

        if (dcVandermondeMatrixRank < dcMonomialSize)
        {
            // We have a singular Vandermonde matrix
            nPointsWithSingularVandermondeMatrix++;

            // If necessary, remove redundant equations/coefficients

            // Number of neighbor points are not enough
            if (dcVandermondeMatrixRank < dcMonomialSize - nENN)
            {
                UMUQWARNING("Number of neighbor points are not enough! Matrix rank = ", dcVandermondeMatrixRank, " < ", dcMonomialSize - nENN);

                if (nENN > 0)
                {
                    VandermondeMatrixTransposeImage.block(0, 0, dcMonomialSize, dcMonomialSize) << VandermondeMatrixTranspose;
                    ExponentialWindowMatrixImage.head(dcMonomialSize) << ExponentialWindowMatrix;

                    // Loop through the rest of nearest neighbors
                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        // Id in the list
                        std::ptrdiff_t const Id = j * nDim;

                        // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                        poly.monomialValue(L1Dist.data() + Id, column);

                        EVectorMapType<RealType> columnV(column + alphamin, dcMonomialSize);

                        // Fill the Vandermonde matrix column by column
                        VandermondeMatrixTransposeImage.block(0, j, dcMonomialSize, 1) << columnV;
                    }

                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        ExponentialWindowMatrixImage(j) = std::exp(-nnDist[j] * byEpsilonSqHalf);
                    }

                    /* 
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMTimage = VandermondeMatrixTransposeImage * EMatrixX<RealType>(ExponentialWindowMatrixImage.asDiagonal());
                    AM = BMTimage * BMTimage.transpose();
                }
                else
                {
                    /* 
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMT = VandermondeMatrixTranspose * EMatrixX<RealType>(ExponentialWindowMatrix.asDiagonal());
                    AM = BMT * BMT.transpose();
                }
            }
            else
            {
                /*
                 * We have enough neighbor points
                 * Remove the columns which causes singularity and replace them
                 * with the new columns from extra neighbor points
                 */

                // Loop through the neighbors
                for (int j = dcVandermondeMatrixRank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t const Id = k * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<RealType> columnV(column + alphamin, dcMonomialSize);

                    // Fill the Vandermonde matrix by the new column
                    VandermondeMatrixTranspose.block(0, l, dcMonomialSize, 1) << columnV;
                }

                for (int j = dcVandermondeMatrixRank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    ExponentialWindowMatrix(l) = std::exp(-nnDist[k] * byEpsilonSqHalf);
                }

                /*
                 * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                 *  {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                 *  {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                 * \f}
                 */
                BMT = VandermondeMatrixTranspose * EMatrixX<RealType>(ExponentialWindowMatrix.asDiagonal());
                AM = BMT * BMT.transpose();
            }

            {
                /*
                 * Two-sided Jacobi SVD decomposition, ensuring optimal reliability and accuracy.
                 * Thin U and V are all we need for (least squares) solving.
                 */
                Eigen::JacobiSVD<EMatrixX<RealType>> svd(AM, Eigen::DecompositionOptions::ComputeThinU | Eigen::DecompositionOptions::ComputeThinV);

                /*
                 * SV contains the least-squares solution of 
                 * \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^RealType({\mathbf x})={\mathbf b} \f$
                 * using the current SVD decomposition of A.
                 */
                SV = svd.solve(RHSB);
            }

            /*!
             * \todo
             * Correct IndexId in the case of SVD. Right now, this is the best I can do
             */
            /*
             * Later I should check on SVD solution and to find out which columns are the
             * Most important one, then I can correct the IndexId order
             */
            if (dcVandermondeMatrixRank < dcMonomialSize - nENN)
            {
                // Loop through the neighbors
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    // Id in the list
                    std::ptrdiff_t const Id = j * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<RealType> columnV(column + alphamin, dcMonomialSize);

                    RealType const expo = std::exp(-nnDist[j] * byEpsilonSq);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + j;
                    dcKernel[IdK] = SV.dot(columnV) * byEpsilonPowerBeta * expo;
                }
            }
            else
            {
                // Loop through the neighbors
                for (int j = 0, m = dcMonomialSize; j < dcMonomialSize; j++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t Id;
                    RealType expo;

                    if (j >= dcVandermondeMatrixRank)
                    {
                        // Id in the list
                        Id = m * nDim;
                        expo = std::exp(-nnDist[m] * byEpsilonSq);
                        m++;
                    }
                    else
                    {
                        Id = l * nDim;
                        expo = std::exp(-nnDist[l] * byEpsilonSq);
                    }

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<RealType> columnV(column + alphamin, dcMonomialSize);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + l;
                    dcKernel[IdK] = SV.dot(columnV) * byEpsilonPowerBeta * expo;
                }

                // Loop through the neighbors
                for (int j = dcVandermondeMatrixRank, m = dcMonomialSize; j < dcMonomialSize; j++, m++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Correct the neighborhood order
                    KNN->IndexSwap(l, m);
                }
            }
        }
        else
        {
            /* 
             * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
             * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
             * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
             * \f}
             */
            BMT = VandermondeMatrixTranspose * EMatrixX<RealType>(ExponentialWindowMatrix.asDiagonal());

            AM = BMT * BMT.transpose();

            // SV contains the solution of \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^RealType({\mathbf x})={\mathbf b} \f$
            SV = AM.lu().solve(RHSB);

            // Loop through the neighbors
            for (int j = 0; j < dcMonomialSize; j++)
            {
                // Id in the list
                std::ptrdiff_t const Id = j * nDim;

                // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                poly.monomialValue(L1Dist.data() + Id, column);

                EVectorMapType<RealType> columnV(column + alphamin, dcMonomialSize);

                RealType const expo = std::exp(-nnDist[j] * byEpsilonSq);

                // Index inside the kernel
                std::ptrdiff_t const IdK = IdM + j;
                dcKernel[IdK] = SV.dot(columnV) * byEpsilonPowerBeta * expo;
            }
        }
    } // Loop over all points

    delete[] column;

    if (nPointsWithSingularVandermondeMatrix > 0)
    {
        UMUQWARNING("There are ", std::to_string(nPointsWithSingularVandermondeMatrix), " query points with singular Vandermonde matrix! (a least-squares solution is used!)");
    }

    return true;
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
bool dcpse<RealType, DistanceType, PolynomialType>::computeWeights(RealType *dataPoints, int const nDataPoints, RealType *queryDataPoints, int const nQueryDataPoints, int *beta, int order, int nENN, RealType ratio)
{
    if (nDataPoints < 1)
    {
        UMUQFAILRETURN("Number of input data points are negative!");
    }

    if (nQueryDataPoints < 1)
    {
        UMUQFAILRETURN("Number of query data points are negative!");
    }

    // Extra check on the order
    order = (order > 0) ? order : 2;
    {
        int *o = Order.data();
        std::fill(o, o + nTerms, order);
    }

    // Extra check on the number of extra nearest neighbors
    nENN = (nENN > 0) ? nENN : 0;

    // Extra check on the ratio
    ratio = (ratio > 0) ? ratio : static_cast<RealType>(1);

    // \f$ |\beta| = \beta_1 + \cdots + \beta_d \f$
    int Beta = std::accumulate(beta, beta + nDim, 0);
    if (Beta == 0)
    {
        UMUQWARNING("Zero order degree derivative gives an approximation! \n If this is an interpolation use the interpolation function!");
    }

    int alphamin = !(Beta & 1);

    // \f$ (-1)^{|\beta|} \f$
    rhscoeff = alphamin ? static_cast<RealType>(1) : -static_cast<RealType>(1);

    // Create an instance of polynomial object with polynomial degree of \f$ |\beta| + r -1 \f$
    PolynomialType poly(nDim, order + Beta - 1);

    /*
     * Get the monomials size
     * \f$ \text{monomialSize} = \left(\begin{matrix} |\beta| + r + d -1 \\ d \end{matrix}\right) - \alpha_{\min} \f$
     */
    dcMonomialSize = poly.monomialsize() - alphamin;

    if (nQueryDataPoints * dcMonomialSize > dcKernelSize)
    {
        dcKernelSize = nQueryDataPoints * dcMonomialSize;
        try
        {
            // Make sure of the correct kernel size
            dcKernel.resize(dcKernelSize);

            h_average.resize(nQueryDataPoints);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        dcKernelSize = nQueryDataPoints * dcMonomialSize;
    }

    if (KNN)
    {
        if (nQueryDataPoints != KNN->numInputdata() || nQueryDataPoints != KNN->numQuerydata())
        {
            try
            {
                /*
                 * Finding K nearest neighbors
                 * The number of points K in the neighborhood of each point
                 * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
                 */
                KNN.reset(new kNearestNeighbor<RealType, DistanceType>(nDataPoints, nQueryDataPoints, nDim, dcMonomialSize + nENN));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
    }
    else
    {
        try
        {
            /*
             * Finding K nearest neighbors
             * The number of points K in the neighborhood of each point
             * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
             */
            KNN.reset(new kNearestNeighbor<RealType, DistanceType>(nDataPoints, nQueryDataPoints, nDim, dcMonomialSize + nENN));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    // If KNN requires covariance
    if (KNN->needsCovariance())
    {
        stats s;
        RealType *Covariance = s.covariance<RealType, RealType>(dataPoints, nDataPoints * nDim, nDim);
        KNN->setCovariance(Covariance);
        delete[] Covariance;
    }

    // Construct a kd-tree index & do nearest neighbors search
    KNN->buildIndex(dataPoints, queryDataPoints);

    /*
     * Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
     * \f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^RealType({\mathbf x})={\mathbf b}  \f$
     */
    EVectorX<RealType> RHSB(dcMonomialSize);
    {
        // Get a pointer to the monomial basis
        int *alpha = poly.monomialBasis();

        for (int i = 0, id = alphamin * nDim; i < dcMonomialSize; i++)
        {
            int maxalpha = 0;
            for (int j = 0; j < nDim; j++, id++)
            {
                maxalpha = std::max(maxalpha, std::abs(alpha[id] - beta[j]));
            }
            if (maxalpha)
            {
                RHSB(i) = RealType{};
            }
            else
            {
                RealType fact = static_cast<RealType>(1);
                std::for_each(beta, beta + nDim, [&](int const b_j) { fact *= factorial<RealType>(b_j); });
                RHSB(i) = rhscoeff * fact;
            }
        }

        /*!
         * \todo
         * Check this again
         */
        /* 
         * At off-particle locations it should be always zero to obtain kernels
         * with a vanishing zeroth-order moment that can be consistently evaluated
         */
        RHSB(0) = RealType{};
    }

    // Total number of nearest neighbors for each point
    int nNN = KNN->numNearestNeighbors();

    // Array for keeping the component-wise L1 distances
    std::vector<RealType> L1Dist(nNN * nDim);

    /*
     * Creating a transpose of the Vandermonde matrix
     * with the size of monomials * monomials \f$  = l \times l \f$
     */
    EMatrixX<RealType> VandermondeMatrixTranspose(dcMonomialSize, dcMonomialSize);
    EMatrixX<RealType> VandermondeMatrixTransposeImage(dcMonomialSize, nNN);

    // Matrix of exponential window function
    EVectorX<RealType> ExponentialWindowMatrix(dcMonomialSize);
    EVectorX<RealType> ExponentialWindowMatrixImage(nNN);

    // Matrix A of a linear system for the kernel coefficients
    EMatrixX<RealType> AM(dcMonomialSize, dcMonomialSize);

    // Matrix B
    EMatrixX<RealType> BMT(dcMonomialSize, dcMonomialSize);
    EMatrixX<RealType> BMTimage(dcMonomialSize, nNN);

    // ${\mathbf a}^RealType({\mathbf x})$ is the column vector of coefficients which is the solution of linear system
    EVectorX<RealType> SV(dcMonomialSize);

    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
    RealType *column = new RealType[dcMonomialSize + alphamin];

    std::vector<int> IndexId(dcMonomialSize);

    // Number of points with singular Vandermonde matrix
    int nPointsWithSingularVandermondeMatrix(0);

    // Loop over all query points
    for (int iQueryDataPoints = 0; iQueryDataPoints < nQueryDataPoints; iQueryDataPoints++)
    {
        std::ptrdiff_t const IdM = iQueryDataPoints * dcMonomialSize;
        std::ptrdiff_t const IdI = iQueryDataPoints * nDim;

        // A pointer to nearest neighbors indices of point iQueryDataPoints
        int *NearestNeighbors = KNN->NearestNeighbors(iQueryDataPoints);

        // A pointer to nearest neighbors square distances from the point iQueryDataPoints
        RealType *nnDist = KNN->NearestNeighborsDistances(iQueryDataPoints);

        /*
         * For each point \f$ {\mathbf x} \f$ we define \f$ \left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x}_p - {\mathbf x} \right\}, \f$
         * as the set of vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points \f${\mathbf x}_p\f$ in the support of \f${\mathbf x}\f$.
         */

        {
            // pointer to query data
            RealType *Idata = queryDataPoints + IdI;

            // \f$ $\left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x} - {\mathbf x}_p \right\} \f$
            for (int j = 0, n = 0; j < nNN; j++)
            {
                std::ptrdiff_t const IdJ = NearestNeighbors[j] * nDim;

                // pointer to dataPoints (neighbors of iQueryDataPoints)
                RealType *Jdata = dataPoints + IdJ;

                for (int d = 0; d < nDim; d++, n++)
                {
                    L1Dist[n] = Idata[d] - Jdata[d];
                }
            }
        }

        // Compute component-wise average neighbor spacing
        RealType h_avg(0);
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](RealType const l_i) { h_avg += std::abs(l_i); });

        // Component-wise average neighbor spacing \f$ h \f$
        h_avg /= static_cast<RealType>(nNN);

        h_average[iQueryDataPoints] = h_avg;

        // Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
        RealType const byEpsilon = ratio / h_avg;
        RealType const byEpsilonSq = byEpsilon * byEpsilon;
        RealType const byEpsilonSqHalf = 0.5 * byEpsilonSq;
        RealType const byEpsilonPowerBeta = std::pow(byEpsilon, Beta);

        // Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](RealType &l_i) { l_i *= byEpsilon; });

        // Loop through the neighbors
        for (int j = 0; j < dcMonomialSize; j++)
        {
            // Id in the L1 distance list
            std::ptrdiff_t const Id = j * nDim;

            // Evaluates a monomial at a point \f$ {\mathbf x} \f$
            poly.monomialValue(L1Dist.data() + Id, column);

            EVectorMapType<RealType> columnV(column + alphamin, dcMonomialSize);

            // Fill the Vandermonde matrix column by column
            VandermondeMatrixTranspose.block(0, j, dcMonomialSize, 1) << columnV;
        }

        for (int j = 0; j < dcMonomialSize; j++)
        {
            ExponentialWindowMatrix(j) = std::exp(-nnDist[j] * byEpsilonSqHalf);
        }

        int dcVandermondeMatrixRank;

        {
            // LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<RealType>> lu(VandermondeMatrixTranspose);

            dcVandermondeMatrixRank = lu.rank();

            if (dcVandermondeMatrixRank < dcMonomialSize && dcVandermondeMatrixRank >= dcMonomialSize - nENN)
            {
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    IndexId[j] = lu.permutationQ().indices()(j);
                }
            }
        }

        if (dcVandermondeMatrixRank < dcMonomialSize)
        {
            // We have a singular Vandermonde matrix
            nPointsWithSingularVandermondeMatrix++;

            // if necessary, remove redundant equations/coefficients

            // Number of neighbor points are not enough
            if (dcVandermondeMatrixRank < dcMonomialSize - nENN)
            {
                UMUQWARNING("Number of neighbor points are not enough! Matrix rank = ", dcVandermondeMatrixRank, " < ", dcMonomialSize - nENN);

                if (nENN > 0)
                {
                    VandermondeMatrixTransposeImage.block(0, 0, dcMonomialSize, dcMonomialSize) << VandermondeMatrixTranspose;
                    ExponentialWindowMatrixImage.head(dcMonomialSize) << ExponentialWindowMatrix;

                    // Loop through the rest of nearest neighbors
                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        // Id in the list
                        std::ptrdiff_t const Id = j * nDim;

                        // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                        poly.monomialValue(L1Dist.data() + Id, column);

                        EVectorMapType<RealType> columnV(column + alphamin, dcMonomialSize);

                        // Fill the Vandermonde matrix column by column
                        VandermondeMatrixTransposeImage.block(0, j, dcMonomialSize, 1) << columnV;
                    }

                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        ExponentialWindowMatrixImage(j) = std::exp(-nnDist[j] * byEpsilonSqHalf);
                    }

                    /*
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMTimage = VandermondeMatrixTransposeImage * EMatrixX<RealType>(ExponentialWindowMatrixImage.asDiagonal());
                    AM = BMTimage * BMTimage.transpose();
                }
                else
                {
                    /* 
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMT = VandermondeMatrixTranspose * EMatrixX<RealType>(ExponentialWindowMatrix.asDiagonal());
                    AM = BMT * BMT.transpose();
                }
            }
            else
            {
                /*
                 * We have enough neighbor points
                 * Remove the columns which causes singularity and replace them
                 * with the new columns from extra neighbor points
                 */

                // Loop through the neighbors
                for (int j = dcVandermondeMatrixRank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t const Id = k * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<RealType> columnV(column + alphamin, dcMonomialSize);

                    // Fill the Vandermonde matrix by the new column
                    VandermondeMatrixTranspose.block(0, l, dcMonomialSize, 1) << columnV;
                }

                for (int j = dcVandermondeMatrixRank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    ExponentialWindowMatrix(l) = std::exp(-nnDist[k] * byEpsilonSqHalf);
                }

                /* 
                 * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                 * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                 * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                 * \f}
                 */
                BMT = VandermondeMatrixTranspose * EMatrixX<RealType>(ExponentialWindowMatrix.asDiagonal());
                AM = BMT * BMT.transpose();
            }

            {
                /*
                 * Two-sided Jacobi SVD decomposition, ensuring optimal reliability and accuracy.
                 * Thin U and V are all we need for (least squares) solving.
                 */
                Eigen::JacobiSVD<EMatrixX<RealType>> svd(AM, Eigen::DecompositionOptions::ComputeThinU | Eigen::DecompositionOptions::ComputeThinV);

                /*
                 * SV contains the least-squares solution of 
                 * \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^RealType({\mathbf x})={\mathbf b} \f$
                 * using the current SVD decomposition of A.
                 */
                SV = svd.solve(RHSB);
            }

            /*!
             * \todo
             * Correct IndexId in the case of SVD. Right now, this is the best I can do
             */
            /*
             * Later I should check on SVD solution and to find out which columns are the
             * Most important one, then I can correct the IndexId order
             */

            if (dcVandermondeMatrixRank < dcMonomialSize - nENN)
            {
                // Loop through the neighbors
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    // Id in the list
                    std::ptrdiff_t const Id = j * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<RealType> columnV(column + alphamin, dcMonomialSize);

                    RealType const expo = std::exp(-nnDist[j] * byEpsilonSq);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + j;
                    dcKernel[IdK] = SV.dot(columnV) * byEpsilonPowerBeta * expo;
                }
            }
            else
            {
                // Loop through the neighbors
                for (int j = 0, m = dcMonomialSize; j < dcMonomialSize; j++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t Id;
                    RealType expo;

                    if (j >= dcVandermondeMatrixRank)
                    {
                        // Id in the list
                        Id = m * nDim;
                        expo = std::exp(-nnDist[m] * byEpsilonSq);
                        m++;
                    }
                    else
                    {
                        Id = l * nDim;
                        expo = std::exp(-nnDist[l] * byEpsilonSq);
                    }

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<RealType> columnV(column + alphamin, dcMonomialSize);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + l;
                    dcKernel[IdK] = SV.dot(columnV) * byEpsilonPowerBeta * expo;
                }

                // Loop through the neighbors
                for (int j = dcVandermondeMatrixRank, m = dcMonomialSize; j < dcMonomialSize; j++, m++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Correct the neighborhood order
                    KNN->IndexSwap(l, m);
                }
            }
        }
        else
        {
            /* 
             * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
             * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
             * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
             * \f}
             */
            BMT = VandermondeMatrixTranspose * EMatrixX<RealType>(ExponentialWindowMatrix.asDiagonal());

            AM = BMT * BMT.transpose();

            // SV contains the solution of \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^RealType({\mathbf x})={\mathbf b} \f$
            SV = AM.lu().solve(RHSB);

            // Loop through the neighbors
            for (int j = 0; j < dcMonomialSize; j++)
            {
                // Id in the list
                std::ptrdiff_t const Id = j * nDim;

                // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                poly.monomialValue(L1Dist.data() + Id, column);

                EVectorMapType<RealType> columnV(column + alphamin, dcMonomialSize);

                RealType const expo = std::exp(-nnDist[j] * byEpsilonSq);

                // Index inside the kernel
                std::ptrdiff_t const IdK = IdM + j;
                dcKernel[IdK] = SV.dot(columnV) * byEpsilonPowerBeta * expo;
            }
        }
    } // Loop over all points

    delete[] column;

    if (nPointsWithSingularVandermondeMatrix > 0)
    {
        UMUQWARNING("There are ", std::to_string(nPointsWithSingularVandermondeMatrix), " query points with singular Vandermonde matrix! (a least-squares solution is used!)");
    }

    return true;
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
bool dcpse<RealType, DistanceType, PolynomialType>::computeInterpolatorWeights(RealType *dataPoints, int const nDataPoints, int order, int nENN, RealType ratio)
{
    if (nDataPoints < 1)
    {
        UMUQFAILRETURN("Number of input data points are negative!");
    }

    // Extra check on the order
    order = (order > 0) ? order : 2;
    std::fill(Order.begin(), Order.end(), order);

    // Extra check on the number of extra nearest neighbors
    nENN = (nENN > 0) ? nENN : 0;

    // Extra check on the ratio
    ratio = (ratio > 0) ? ratio : static_cast<RealType>(1);

    // Create an instance of a polynomial object with polynomial degree of \f$ |\beta| + r - 1 \f$
    PolynomialType poly(nDim, order - 1);

    /* 
     * Get the monomials size
     * \f$ \text{monomialSize} = \left(\begin{matrix} r + d -1 \\ d \end{matrix}\right) \f$
     */
    dcMonomialSize = poly.monomialsize();

    if (nDataPoints * dcMonomialSize > dcKernelSize)
    {
        dcKernelSize = nDataPoints * dcMonomialSize;
        try
        {
            // Make sure of the correct kernel size
            dcKernel.resize(dcKernelSize);

            h_average.resize(nDataPoints);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        dcKernelSize = nDataPoints * dcMonomialSize;
    }

    if (KNN)
    {
        if (nDataPoints != KNN->numInputdata() || nDataPoints != KNN->numQuerydata())
        {
            try
            {
                /* 
                 * Finding K nearest neighbors
                 * The number of points K in the neighborhood of each point
                 * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
                 */
                KNN.reset(new kNearestNeighbor<RealType, DistanceType>(nDataPoints, nDataPoints, nDim, dcMonomialSize + nENN));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
    }
    else
    {
        try
        {
            /* 
             * Finding K nearest neighbors
             * The number of points K in the neighborhood of each point
             * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
             */
            KNN.reset(new kNearestNeighbor<RealType, DistanceType>(nDataPoints, nDataPoints, nDim, dcMonomialSize + nENN));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    // If KNN requires covariance
    if (KNN->needsCovariance())
    {
        stats s;
        RealType *Covariance = s.covariance<RealType, RealType>(dataPoints, nDataPoints * nDim, nDim);
        KNN->setCovariance(Covariance);
        delete[] Covariance;
    }

    // Construct a kd-tree index & do nearest neighbors search
    KNN->buildIndex(dataPoints);

    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
    RealType *column = new RealType[dcMonomialSize];

    // Evaluates a monomial at a point \f$ {\mathbf x} = 0. \f$
    poly.monomialValue(std::vector<RealType>(nDim, RealType{}).data(), column);

    /*
     * Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
     * \f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^RealType({\mathbf x})={\mathbf b}  \f$
     */
    EVectorX<RealType> RHSB0 = EVectorMapType<RealType>(column, dcMonomialSize);
    EVectorX<RealType> RHSB(dcMonomialSize);

    // Total number of nearest neighbors for each point
    int nNN = KNN->numNearestNeighbors();

    /*
     * Creating a transpose of the Vandermonde matrix
     * with the size of monomials * monomials \f$  = l \times l \f$
     */
    EMatrixX<RealType> VandermondeMatrixTranspose(dcMonomialSize, dcMonomialSize);
    EMatrixX<RealType> VandermondeMatrixTransposeImage(dcMonomialSize, nNN);

    // Matrix of exponential window function
    EVectorX<RealType> ExponentialWindowMatrix(dcMonomialSize);
    EVectorX<RealType> ExponentialWindowMatrixImage(nNN);

    EVectorX<RealType> columnL(dcMonomialSize);

    // Matrix A of a linear system for the kernel coefficients
    EMatrixX<RealType> AM(dcMonomialSize, dcMonomialSize);

    // Matrix B^RealType
    EMatrixX<RealType> BMT(dcMonomialSize, dcMonomialSize);
    EMatrixX<RealType> BMTimage(dcMonomialSize, nNN);

    // ${\mathbf a}^RealType({\mathbf x})$ is the column vector of coefficients which is the solution of linear system
    EVectorX<RealType> SV(dcMonomialSize);

    // Array for keeping the component-wise L1 distances
    std::vector<RealType> L1Dist(nNN * nDim);

    // Array to kepp indexing
    std::vector<int> IndexId(dcMonomialSize);

    // Number of points with singular Vandermonde matrix
    int nPointsWithSingularVandermondeMatrix(0);

    // Loop over all query points
    for (int iQueryDataPoints = 0; iQueryDataPoints < nDataPoints; iQueryDataPoints++)
    {
        // Index inside kernel
        std::ptrdiff_t const IdM = iQueryDataPoints * dcMonomialSize;

        // Index in dataPoints array
        std::ptrdiff_t const IdI = iQueryDataPoints * nDim;

        // A pointer to nearest neighbors indices of point iQueryDataPoints
        int *NearestNeighbors = KNN->NearestNeighbors(iQueryDataPoints);

        // A pointer to nearest neighbors square distances from the point iQueryDataPoints
        RealType *nnDist = KNN->NearestNeighborsDistances(iQueryDataPoints);

        /*
         * For each point \f$ {\mathbf x} \f$ we define 
         * \f$ \left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x}_p - {\mathbf x} \right\}, \f$
         * as the set of vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points 
         * \f${\mathbf x}_p\f$ in the support of \f${\mathbf x}\f$.
         */
        {
            // pointer to query data
            RealType *Idata = dataPoints + IdI;

            // \f$ $\left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x} - {\mathbf x}_p \right\} \f$
            for (int j = 0, n = 0; j < nNN; j++)
            {
                // Neighbor index in dataPoints array
                std::ptrdiff_t const IdJ = NearestNeighbors[j] * nDim;

                // pointer to dataPoints (neighbors of iQueryDataPoints)
                RealType *Jdata = dataPoints + IdJ;

                for (int d = 0; d < nDim; d++, n++)
                {
                    L1Dist[n] = Idata[d] - Jdata[d];
                }
            }
        }

        // Compute component-wise average neighbor spacing
        RealType h_avg(0);
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](RealType const l_i) { h_avg += std::abs(l_i); });

        // Component-wise average neighbor spacing \f$ h \f$
        h_avg /= static_cast<RealType>(nNN);

        h_average[iQueryDataPoints] = h_avg;

        // Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
        RealType const byEpsilon = ratio / h_avg;
        RealType const byEpsilonSq = byEpsilon * byEpsilon;
        RealType const byEpsilonSqHalf = 0.5 * byEpsilonSq;

        // Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](RealType &l_i) { l_i *= byEpsilon; });

        // Use the correct RHS for each point
        RHSB = RHSB0;

        // Loop through the neighbors
        for (int j = 0; j < dcMonomialSize; j++)
        {
            // Id in the L1 distance list
            std::ptrdiff_t const Id = j * nDim;

            // Evaluates a monomial at a point \f$ {\mathbf x} \f$
            poly.monomialValue(L1Dist.data() + Id, column);

            EVectorMapType<RealType> columnV(column, dcMonomialSize);

            // Fill the Vandermonde matrix column by column
            VandermondeMatrixTranspose.block(0, j, dcMonomialSize, 1) << columnV;
        }

        for (int j = 0; j < dcMonomialSize; j++)
        {
            ExponentialWindowMatrix(j) = std::exp(-nnDist[j] * byEpsilonSqHalf);
        }

        int dcVandermondeMatrixRank;

        {
            // LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<RealType>> lu(VandermondeMatrixTranspose);

            dcVandermondeMatrixRank = lu.rank();

            if (dcVandermondeMatrixRank < dcMonomialSize && dcVandermondeMatrixRank >= dcMonomialSize - nENN)
            {
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    IndexId[j] = lu.permutationQ().indices()(j);
                }
            }
        }

        if (dcVandermondeMatrixRank < dcMonomialSize)
        {
            // We have a singular Vandermonde matrix
            nPointsWithSingularVandermondeMatrix++;

            // if necessary, remove redundant equations/coefficients

            // Number of neighbor points are not enough
            if (dcVandermondeMatrixRank < dcMonomialSize - nENN)
            {
                UMUQWARNING("Number of neighbor points are not enough! Matrix rank = ", dcVandermondeMatrixRank, " < ", dcMonomialSize - nENN);

                if (nENN > 0)
                {
                    VandermondeMatrixTransposeImage.block(0, 0, dcMonomialSize, dcMonomialSize) << VandermondeMatrixTranspose;
                    ExponentialWindowMatrixImage.head(dcMonomialSize) << ExponentialWindowMatrix;

                    // Loop through the rest of nearest neighbors
                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        // Id in the list
                        std::ptrdiff_t const Id = j * nDim;

                        // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                        poly.monomialValue(L1Dist.data() + Id, column);

                        EVectorMapType<RealType> columnV(column, dcMonomialSize);

                        // Fill the Vandermonde matrix column by column
                        VandermondeMatrixTransposeImage.block(0, j, dcMonomialSize, 1) << columnV;
                    }

                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        ExponentialWindowMatrixImage(j) = std::exp(-nnDist[j] * byEpsilonSqHalf);
                    }

                    /* 
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMTimage = VandermondeMatrixTransposeImage * EMatrixX<RealType>(ExponentialWindowMatrixImage.asDiagonal());
                    AM = BMTimage * BMTimage.transpose();
                }
                else
                {
                    /* 
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMT = VandermondeMatrixTranspose * EMatrixX<RealType>(ExponentialWindowMatrix.asDiagonal());
                    AM = BMT * BMT.transpose();
                }
            }
            else
            {
                /*
                 * We have enough neighbor points
                 * Remove the columns which causes singularity and replace them
                 * with the new columns from extra neighbor points
                 */

                // Loop through the neighbors
                for (int j = dcVandermondeMatrixRank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t const Id = k * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<RealType> columnV(column, dcMonomialSize);

                    // Get the column l which causes singularity
                    columnL << VandermondeMatrixTranspose.block(0, l, dcMonomialSize, 1);

                    // Fill the Vandermonde matrix by the new column
                    VandermondeMatrixTranspose.block(0, l, dcMonomialSize, 1) << columnV;
                }

                for (int j = dcVandermondeMatrixRank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    ExponentialWindowMatrix(l) = std::exp(-nnDist[k] * byEpsilonSqHalf);
                }

                /* 
                 * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                 * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                 * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                 * \f}
                 */
                BMT = VandermondeMatrixTranspose * EMatrixX<RealType>(ExponentialWindowMatrix.asDiagonal());
                AM = BMT * BMT.transpose();
            }

            {
                /*
                 * Two-sided Jacobi SVD decomposition, ensuring optimal reliability and accuracy.
                 * Thin U and V are all we need for (least squares) solving.
                 */
                Eigen::JacobiSVD<EMatrixX<RealType>> svd(AM, Eigen::DecompositionOptions::ComputeThinU | Eigen::DecompositionOptions::ComputeThinV);

                /*
                 * SV contains the least-squares solution of 
                 * \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^RealType({\mathbf x})={\mathbf b} \f$
                 * using the current SVD decomposition of A.
                 */
                SV = svd.solve(RHSB);
            }

            /*!
             * \todo
             * Correct IndexId in the case of SVD. Right now, this is the best I can do
             */
            /*
             * Later I should check on SVD solution and to find out which columns are the
             * Most important one, then I can correct the IndexId order
             */

            if (dcVandermondeMatrixRank < dcMonomialSize - nENN)
            {
                // Loop through the neighbors
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    // Id in the list
                    std::ptrdiff_t const Id = j * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<RealType> columnV(column, dcMonomialSize);

                    RealType const expo = std::exp(-nnDist[j] * byEpsilonSq);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + j;
                    dcKernel[IdK] = SV.dot(columnV) * expo;
                }
            }
            else
            {
                // Loop through the neighbors
                for (int j = 0, m = dcMonomialSize; j < dcMonomialSize; j++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t Id;
                    RealType expo;

                    if (j >= dcVandermondeMatrixRank)
                    {
                        // Id in the list
                        Id = m * nDim;
                        expo = std::exp(-nnDist[m] * byEpsilonSq);
                        m++;
                    }
                    else
                    {
                        Id = l * nDim;
                        expo = std::exp(-nnDist[l] * byEpsilonSq);
                    }

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<RealType> columnV(column, dcMonomialSize);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + l;
                    dcKernel[IdK] = SV.dot(columnV) * expo;
                }

                // Loop through the neighbors
                for (int j = dcVandermondeMatrixRank, m = dcMonomialSize; j < dcMonomialSize; j++, m++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Correct the neighborhood order
                    KNN->IndexSwap(l, m);
                }
            }
        }
        else
        {
            /* 
             * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
             * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
             * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
             * \f}
             */
            BMT = VandermondeMatrixTranspose * EMatrixX<RealType>(ExponentialWindowMatrix.asDiagonal());

            AM = BMT * BMT.transpose();

            // SV contains the solution of \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^RealType({\mathbf x})={\mathbf b} \f$
            SV = AM.lu().solve(RHSB);

            // Loop through the neighbors
            for (int j = 0; j < dcMonomialSize; j++)
            {
                // Id in the list
                std::ptrdiff_t const Id = j * nDim;

                // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                poly.monomialValue(L1Dist.data() + Id, column);

                EVectorMapType<RealType> columnV(column, dcMonomialSize);

                RealType const expo = std::exp(-nnDist[j] * byEpsilonSq);

                // Index inside the kernel
                std::ptrdiff_t const IdK = IdM + j;
                dcKernel[IdK] = SV.dot(columnV) * expo;
            }
        }

    } // Loop over all points

    delete[] column;

    if (nPointsWithSingularVandermondeMatrix > 0)
    {
        UMUQWARNING("There are ", std::to_string(nPointsWithSingularVandermondeMatrix), " query points with singular Vandermonde matrix! (a least-squares solution is used!)");
    }

    return true;
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
bool dcpse<RealType, DistanceType, PolynomialType>::computeInterpolatorWeights(RealType *dataPoints, int const nDataPoints, RealType *queryDataPoints, int const nQueryDataPoints, int order, int nENN, RealType ratio)
{
    if (nDataPoints < 1)
    {
        UMUQFAILRETURN("Number of input data points are negative!");
    }

    if (nQueryDataPoints < 1)
    {
        UMUQFAILRETURN("Number of query data points are negative!");
    }

    // Extra check on the order
    order = (order > 0) ? order : 2;
    std::fill(Order.begin(), Order.end(), order);

    // Extra check on the number of extra nearest neighbors
    nENN = (nENN > 0) ? nENN : 0;

    // Extra check on the ratio
    ratio = (ratio > 0) ? ratio : static_cast<RealType>(1);

    // Create an instance of a polynomial object with polynomial degree of \f$ |\beta| + r - 1 \f$
    PolynomialType poly(nDim, order - 1);

    /* 
     * Get the monomials size
     * \f$ \text{monomialSize} = \left(\begin{matrix} r + d -1 \\ d \end{matrix}\right) \f$
     */
    dcMonomialSize = poly.monomialsize();

    if (nQueryDataPoints * dcMonomialSize > dcKernelSize)
    {
        dcKernelSize = nQueryDataPoints * dcMonomialSize;
        try
        {
            // Make sure of the correct kernel size
            dcKernel.resize(dcKernelSize);

            h_average.resize(nQueryDataPoints);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        dcKernelSize = nQueryDataPoints * dcMonomialSize;
    }

    if (KNN)
    {
        if (nQueryDataPoints != KNN->numInputdata() || nQueryDataPoints != KNN->numQuerydata())
        {
            try
            {
                /* 
                 * Finding K nearest neighbors
                 * The number of points K in the neighborhood of each point
                 * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
                 */
                KNN.reset(new kNearestNeighbor<RealType, DistanceType>(nDataPoints, nQueryDataPoints, nDim, dcMonomialSize + nENN));
            }
            catch (...)
            {
                UMUQFAILRETURN("Failed to allocate memory!");
            }
        }
    }
    else
    {
        try
        {
            /* 
             * Finding K nearest neighbors
             * The number of points K in the neighborhood of each point
             * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
             */
            KNN.reset(new kNearestNeighbor<RealType, DistanceType>(nDataPoints, nQueryDataPoints, nDim, dcMonomialSize + nENN));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    // Vector of all dataPoints points' distances from their closest nearest neighbor
    RealType *idataminDist = nullptr;

    {
        RealType *Covariance = nullptr;

        // If KNN requires covariance
        if (KNN->needsCovariance())
        {
            stats s;
            Covariance = s.covariance<RealType, RealType>(dataPoints, nDataPoints * nDim, nDim);
            KNN->setCovariance(Covariance);
        }

        // Construct a kd-tree index & do nearest neighbors search
        KNN->buildIndex(dataPoints, queryDataPoints);

        {
            // Finding only one nearest neighbor for the input data points
            kNearestNeighbor<RealType, DistanceType> KNN1(nDataPoints, nDim, 1);

            if (KNN1.needsCovariance())
            {
                KNN1.setCovariance(Covariance);

                delete[] Covariance;
                Covariance = nullptr;
            }

            // Construct a kd-tree index & do nearest neighbors search
            KNN1.buildIndex(dataPoints);

            idataminDist = KNN1.minDist();

            if (idataminDist == nullptr)
            {
                UMUQFAILRETURN("Failed to find any neighbor!");
            }
        }
    }

    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
    RealType *column = new RealType[dcMonomialSize];

    // Evaluates a monomial at a point \f$ {\mathbf x} = 0. \f$
    poly.monomialValue(std::vector<RealType>(nDim, RealType{}).data(), column);

    /*
     * Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
     * \f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^RealType({\mathbf x})={\mathbf b}  \f$
     */
    EVectorX<RealType> RHSB0 = EVectorMapType<RealType>(column, dcMonomialSize);
    EVectorX<RealType> RHSB(dcMonomialSize);

    // Total number of nearest neighbors for each point
    int nNN = KNN->numNearestNeighbors();

    /*
     * Creating a transpose of the Vandermonde matrix
     * with the size of monomials * monomials \f$  = l \times l \f$
     */
    EMatrixX<RealType> VandermondeMatrixTranspose(dcMonomialSize, dcMonomialSize);
    EMatrixX<RealType> VandermondeMatrixTransposeImage(dcMonomialSize, nNN);

    // Matrix of exponential window function
    EVectorX<RealType> ExponentialWindowMatrix(dcMonomialSize);
    EVectorX<RealType> ExponentialWindowMatrixImage(nNN);

    EVectorX<RealType> columnL(dcMonomialSize);

    // Matrix A of a linear system for the kernel coefficients
    EMatrixX<RealType> AM(dcMonomialSize, dcMonomialSize);

    // Matrix B^RealType
    EMatrixX<RealType> BMT(dcMonomialSize, dcMonomialSize);
    EMatrixX<RealType> BMTimage(dcMonomialSize, nNN);

    // ${\mathbf a}^RealType({\mathbf x})$ is the column vector of coefficients which is the solution of linear system
    EVectorX<RealType> SV(dcMonomialSize);

    // Array for keeping the component-wise L1 distances
    std::vector<RealType> L1Dist(nNN * nDim);

    // Array to keep indexing
    std::vector<int> IndexId(dcMonomialSize);

    // Primitive (quartic spline) object
    quartic_spline<RealType> q;

    // Number of points with singular Vandermonde matrix
    int nPointsWithSingularVandermondeMatrix(0);

    // Loop over all query points
    for (int iQueryDataPoints = 0; iQueryDataPoints < nQueryDataPoints; iQueryDataPoints++)
    {
        // Index inside kernel
        std::ptrdiff_t const IdM = iQueryDataPoints * dcMonomialSize;

        // Index in queryDataPoints array
        std::ptrdiff_t const IdI = iQueryDataPoints * nDim;

        // A pointer to nearest neighbors indices of point iQueryDataPoints
        int *NearestNeighbors = KNN->NearestNeighbors(iQueryDataPoints);

        // A pointer to nearest neighbors square distances from the point iQueryDataPoints
        RealType *nnDist = KNN->NearestNeighborsDistances(iQueryDataPoints);

        /*
         * For each point \f$ {\mathbf x} \f$ we define 
         * \f$ \left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x}_p - {\mathbf x} \right\}, \f$
         * as the set of vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points 
         * \f${\mathbf x}_p\f$ in the support of \f${\mathbf x}\f$.
         */
        {
            // pointer to query data
            RealType *Idata = queryDataPoints + IdI;

            // \f$ $\left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x} - {\mathbf x}_p \right\} \f$
            for (int j = 0, n = 0; j < nNN; j++)
            {
                // Neighbor index in dataPoints array
                std::ptrdiff_t const IdJ = NearestNeighbors[j] * nDim;

                // pointer to dataPoints (neighbors of iQueryDataPoints)
                RealType *Jdata = dataPoints + IdJ;

                for (int d = 0; d < nDim; d++, n++)
                {
                    L1Dist[n] = Idata[d] - Jdata[d];
                }
            }
        }

        // Compute component-wise average neighbor spacing
        RealType h_avg(0);
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](RealType const l_i) { h_avg += std::abs(l_i); });

        // Component-wise average neighbor spacing \f$ h \f$
        h_avg /= static_cast<RealType>(nNN);

        h_average[iQueryDataPoints] = h_avg;

        // Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
        RealType const byEpsilon = ratio / h_avg;
        RealType const byEpsilonSq = byEpsilon * byEpsilon;
        RealType const byEpsilonSqHalf = 0.5 * byEpsilonSq;

        // Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](RealType &l_i) { l_i *= byEpsilon; });

        // Use the correct RHS for each point
        RHSB = RHSB0;

        // Loop through the neighbors
        for (int j = 0; j < dcMonomialSize; j++)
        {
            // Id in the L1 distance list
            std::ptrdiff_t const Id = j * nDim;

            // Evaluates a monomial at a point \f$ {\mathbf x} \f$
            poly.monomialValue(L1Dist.data() + Id, column);

            EVectorMapType<RealType> columnV(column, dcMonomialSize);

            // Fill the Vandermonde matrix column by column
            VandermondeMatrixTranspose.block(0, j, dcMonomialSize, 1) << columnV;

            // Neighbor point number
            int const IdJ = NearestNeighbors[j];

            /* 
             * Using a smooth correction function that satisfies
             * \f$ {\mathbf F} \left(\frac{{\mathbf x}_p-{\mathbf x}_q}{c({\mathbf x}_q)} \right) =\delta_{pq} \f$
             * Choose \f$ c({\mathbf x}) \f$ such that it is smaller than the distance
             * between the point and its nearest neighbors
             */
            RealType s = std::sqrt(nnDist[j]) / (0.9 * std::sqrt(idataminDist[IdJ]));

            // Compute the kernel value at the point
            RealType const dckernelV = q.f(&s);

            /* 
             * Assemble the right hand side
             * 
             * \f[
             * {\mathbf b}={\mathbf P}({\mathbf x}) |_{{\mathbf x}=0} - 
             * \sum_{p} {\mathbf P}{\left(\frac{{\mathbf x}-{\mathbf x}_p}{\epsilon({\mathbf x})}\right)} 
             * {\mathbf C}\left(\frac{{\mathbf x}-{\mathbf x}_p}{c({\mathbf x}_p)} \right) 
             * \f]
             * 
             */
            RHSB -= dckernelV * columnV;

            // Index inside the kernel
            std::ptrdiff_t const IdK = IdM + j;
            dcKernel[IdK] = dckernelV;
        }

        for (int j = 0; j < dcMonomialSize; j++)
        {
            ExponentialWindowMatrix(j) = std::exp(-nnDist[j] * byEpsilonSqHalf);
        }

        int dcVandermondeMatrixRank;

        {
            // LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<RealType>> lu(VandermondeMatrixTranspose);

            dcVandermondeMatrixRank = lu.rank();

            if (dcVandermondeMatrixRank < dcMonomialSize && dcVandermondeMatrixRank >= dcMonomialSize - nENN)
            {
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    IndexId[j] = lu.permutationQ().indices()(j);
                }
            }
        }

        if (dcVandermondeMatrixRank < dcMonomialSize)
        {
            // We have a singular Vandermonde matrix
            nPointsWithSingularVandermondeMatrix++;

            // if necessary, remove redundant equations/coefficients

            // Number of neighbor points are not enough
            if (dcVandermondeMatrixRank < dcMonomialSize - nENN)
            {
                UMUQWARNING("Number of neighbor points are not enough! Matrix rank = ", dcVandermondeMatrixRank, " < ", dcMonomialSize - nENN);

                if (nENN > 0)
                {
                    VandermondeMatrixTransposeImage.block(0, 0, dcMonomialSize, dcMonomialSize) << VandermondeMatrixTranspose;
                    ExponentialWindowMatrixImage.head(dcMonomialSize) << ExponentialWindowMatrix;

                    // Loop through the rest of nearest neighbors
                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        // Id in the list
                        std::ptrdiff_t const Id = j * nDim;

                        // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                        poly.monomialValue(L1Dist.data() + Id, column);

                        EVectorMapType<RealType> columnV(column, dcMonomialSize);

                        // Fill the Vandermonde matrix column by column
                        VandermondeMatrixTransposeImage.block(0, j, dcMonomialSize, 1) << columnV;

                        // Neighbor point number
                        int const IdJ = NearestNeighbors[j];

                        /*
                         * Using a smooth correction function that satisfies
                         * \f$ {\mathbf F} \left(\frac{{\mathbf x}_p-{\mathbf x}_q}{c({\mathbf x}_q)} \right) =\delta_{pq} \f$
                         * Choose \f$ c({\mathbf x}) \f$ such that it is smaller than the distance
                         * between the point and its nearest neighbors
                         */
                        RealType s = std::sqrt(nnDist[j]) / (0.9 * std::sqrt(idataminDist[IdJ]));

                        // Compute the kernel value at the point
                        RealType const dckernelV = q.f(&s);

                        /*
                         * Assemble the right hand side<br>
                         * \f$
                         *  {\mathbf b}={\mathbf P}({\mathbf x}) |_{{\mathbf x}=0} - 
                         *  \sum_{p} {\mathbf P}{\left(\frac{{\mathbf x}-{\mathbf x}_p}{\epsilon({\mathbf x})}\right)} 
                         *  {\mathbf C}\left(\frac{{\mathbf x}-{\mathbf x}_p}{c({\mathbf x}_p)} \right) 
                         * \f$
                         */
                        RHSB -= dckernelV * columnV;
                    }

                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        ExponentialWindowMatrixImage(j) = std::exp(-nnDist[j] * byEpsilonSqHalf);
                    }

                    /* 
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMTimage = VandermondeMatrixTransposeImage * EMatrixX<RealType>(ExponentialWindowMatrixImage.asDiagonal());
                    AM = BMTimage * BMTimage.transpose();
                }
                else
                {
                    /* 
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMT = VandermondeMatrixTranspose * EMatrixX<RealType>(ExponentialWindowMatrix.asDiagonal());
                    AM = BMT * BMT.transpose();
                }
            }
            else
            {
                /*
                 * We have enough neighbor points
                 * Remove the columns which causes singularity and replace them
                 * with the new columns from extra neighbor points
                 */

                // Loop through the neighbors
                for (int j = dcVandermondeMatrixRank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t const Id = k * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<RealType> columnV(column, dcMonomialSize);

                    // Get the column l which causes singularity
                    columnL << VandermondeMatrixTranspose.block(0, l, dcMonomialSize, 1);

                    // Fill the Vandermonde matrix by the new column
                    VandermondeMatrixTranspose.block(0, l, dcMonomialSize, 1) << columnV;

                    // Neighbor point number
                    int const IdJ = NearestNeighbors[k];

                    /* 
                     * Using a smooth correction function that satisfies
                     * \f$ {\mathbf F} \left(\frac{{\mathbf x}_p-{\mathbf x}_q}{c({\mathbf x}_q)} \right) =\delta_{pq} \f$
                     * Choose \f$ c({\mathbf x}) \f$ such that it is smaller than the distance
                     * between the point and its nearest neighbors
                     */
                    RealType s = std::sqrt(nnDist[k]) / (0.9 * std::sqrt(idataminDist[IdJ]));

                    // Compute the kernel value at the point IdK
                    RealType dckernelV = q.f(&s);

                    // Index of the column l inside the kernel
                    std::ptrdiff_t const IdK = IdM + j;

                    dcKernel[IdK] = dckernelV;

                    /*
                     * Assemble the right hand side
                     * 
                     * \f[
                     *  {\mathbf b}={\mathbf P}({\mathbf x}) |_{{\mathbf x}=0} - \sum_{p} 
                     * {\mathbf P}{\left(\frac{{\mathbf x}-{\mathbf x}_p}{\epsilon({\mathbf x})}\right)} 
                     * {\mathbf C}\left(\frac{{\mathbf x}-{\mathbf x}_p}{c({\mathbf x}_p)} \right) 
                     * \f]
                     */
                    RHSB -= dckernelV * columnV;

                    // Neighbor point number of point l which causes singularity
                    int const IdJL = NearestNeighbors[l];
                    s = std::sqrt(nnDist[l]) / (0.9 * std::sqrt(idataminDist[IdJL]));
                    dckernelV = q.f(&s);
                    RHSB += dckernelV * columnL;
                }

                for (int j = dcVandermondeMatrixRank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    ExponentialWindowMatrix(l) = std::exp(-nnDist[k] * byEpsilonSqHalf);
                }

                /* 
                 * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                 * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                 * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1} 
                 * \f}
                 */
                BMT = VandermondeMatrixTranspose * EMatrixX<RealType>(ExponentialWindowMatrix.asDiagonal());
                AM = BMT * BMT.transpose();
            }

            {
                /*
                 * Two-sided Jacobi SVD decomposition, ensuring optimal reliability and accuracy.
                 * Thin U and V are all we need for (least squares) solving.
                 */
                Eigen::JacobiSVD<EMatrixX<RealType>> svd(AM, Eigen::DecompositionOptions::ComputeThinU | Eigen::DecompositionOptions::ComputeThinV);

                /*
                 * SV contains the least-squares solution of 
                 * \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^RealType({\mathbf x})={\mathbf b} \f$
                 * using the current SVD decomposition of A.
                 */
                SV = svd.solve(RHSB);
            }

            /*!
             * \todo
             * Correct IndexId in the case of SVD. Right now, this is the best I can do
             */
            /*
             * Later I should check on SVD solution and to find out which columns are the
             * Most important one, then I can correct the IndexId order
             */

            if (dcVandermondeMatrixRank < dcMonomialSize - nENN)
            {
                // Loop through the neighbors
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    // Id in the list
                    std::ptrdiff_t const Id = j * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<RealType> columnV(column, dcMonomialSize);

                    RealType const expo = std::exp(-nnDist[j] * byEpsilonSq);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + j;
                    dcKernel[IdK] += SV.dot(columnV) * expo;
                }
            }
            else
            {
                // Loop through the neighbors
                for (int j = 0, m = dcMonomialSize; j < dcMonomialSize; j++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t Id;
                    RealType expo;

                    if (j >= dcVandermondeMatrixRank)
                    {
                        // Id in the list
                        Id = m * nDim;
                        expo = std::exp(-nnDist[m] * byEpsilonSq);
                        m++;
                    }
                    else
                    {
                        Id = l * nDim;
                        expo = std::exp(-nnDist[l] * byEpsilonSq);
                    }

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<RealType> columnV(column, dcMonomialSize);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + l;
                    dcKernel[IdK] += SV.dot(columnV) * expo;
                }

                // Loop through the neighbors
                for (int j = dcVandermondeMatrixRank, m = dcMonomialSize; j < dcMonomialSize; j++, m++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Correct the neighborhood order
                    KNN->IndexSwap(l, m);
                }
            }
        }
        else
        {
            /* 
             * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^RealType ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
             * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
             * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1} 
             * \f}
             */
            BMT = VandermondeMatrixTranspose * EMatrixX<RealType>(ExponentialWindowMatrix.asDiagonal());

            AM = BMT * BMT.transpose();

            // SV contains the solution of \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^RealType({\mathbf x})={\mathbf b} \f$
            SV = AM.lu().solve(RHSB);

            // Loop through the neighbors
            for (int j = 0; j < dcMonomialSize; j++)
            {
                // Id in the list
                std::ptrdiff_t const Id = j * nDim;

                // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                poly.monomialValue(L1Dist.data() + Id, column);

                EVectorMapType<RealType> columnV(column, dcMonomialSize);

                RealType const expo = std::exp(-nnDist[j] * byEpsilonSq);

                // Index inside the kernel
                std::ptrdiff_t const IdK = IdM + j;
                dcKernel[IdK] += SV.dot(columnV) * expo;
            }
        }

    } // Loop over all points

    delete[] column;
    delete[] idataminDist;

    if (nPointsWithSingularVandermondeMatrix > 0)
    {
        UMUQWARNING("There are ", std::to_string(nPointsWithSingularVandermondeMatrix), " query points with singular Vandermonde matrix! (a least-squares solution is used!)");
    }

    return true;
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
bool dcpse<RealType, DistanceType, PolynomialType>::compute(RealType *dataFunctionValues, int const nDataPoints, RealType *queryFunctionValues, int const nQueryDataPoints)
{
    if (KNN->numInputdata() != nDataPoints)
    {
        UMUQFAILRETURN("Input data does not match with previously computed weights!");
    }
    if (KNN->numQuerydata() != nQueryDataPoints)
    {
        UMUQFAILRETURN("Query data does not match with previously computed weights!");
    }
    if (dcKernelSize != nQueryDataPoints * dcMonomialSize)
    {
        UMUQFAILRETURN("Previously computed weights does not match with this query data!");
    }

    if (queryFunctionValues == nullptr)
    {
        try
        {
            queryFunctionValues = new RealType[nQueryDataPoints];
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
bool dcpse<RealType, DistanceType, PolynomialType>::interpolate(RealType const *dataFunctionValues, int const nDataPoints, RealType *queryFunctionValues, int const nQueryDataPoints)
{
    if (KNN->numInputdata() != nDataPoints)
    {
        UMUQFAILRETURN("Input data does not match with previously computed weights!");
    }
    if (KNN->numQuerydata() != nQueryDataPoints)
    {
        UMUQFAILRETURN("Query data does not match with previously computed weights!");
    }
    if (dcKernelSize != nQueryDataPoints * dcMonomialSize)
    {
        UMUQFAILRETURN("Previously computed weights does not match with this query data!");
    }

    if (queryFunctionValues == nullptr)
    {
        UMUQFAILRETURN("Memory is not assigned for pointer to the query function values!");
    }

    // Loop over all query points
    for (int iQueryDataPoints = 0; iQueryDataPoints < nQueryDataPoints; iQueryDataPoints++)
    {
        // A pointer to nearest neighbors indices of point iQueryDataPoints
        int *NearestNeighbors = KNN->NearestNeighbors(iQueryDataPoints);

        int IdI = iQueryDataPoints * dcMonomialSize;

        RealType sum(0);

        // Loop through the neighbors
        for (int j = 0; j < dcMonomialSize; j++, IdI++)
        {
            int const IdJ = NearestNeighbors[j];
            sum += dcKernel[IdI] * dataFunctionValues[IdJ];
        }
        queryFunctionValues[iQueryDataPoints] = sum;
    }

    return true;
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
inline RealType *dcpse<RealType, DistanceType, PolynomialType>::neighborhoodKernel(int const index) const
{
    return dcKernel.data() + index * dcMonomialSize;
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
inline RealType *dcpse<RealType, DistanceType, PolynomialType>::neighborhoodKernel() const
{
    return dcKernel.data();
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
inline int dcpse<RealType, DistanceType, PolynomialType>::neighborhoodKernelSize() const
{
    return dcMonomialSize;
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
inline int dcpse<RealType, DistanceType, PolynomialType>::orderofAccuracy(int const index) const
{
    return Order[index];
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
inline void dcpse<RealType, DistanceType, PolynomialType>::printInfo() const
{
    for (int i = 0; i < nTerms; i++)
    {
        std::cout << Order[i] << (Order[i] % 10 == 1 ? "st " : Order[i] % 10 == 2 ? "nd " : Order[i] % 10 == 3 ? "rd " : "th ") << "order DC-PSE kernel uses \n"
                  << neighborhoodKernelSize() << " points in the neighborhood of each query point." << std::endl;
    }
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
inline RealType dcpse<RealType, DistanceType, PolynomialType>::averageSpace(int const index) const
{
    return h_average[index];
}

template <typename RealType, umuq::NeighborDistanceTypes DistanceType, class PolynomialType>
inline RealType *dcpse<RealType, DistanceType, PolynomialType>::averageSpace() const
{
    return h_average.data();
}

} // namespace umuq

#endif // UMUQ_DCPSE

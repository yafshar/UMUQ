#ifndef UMUQ_DCPSE_H
#define UMUQ_DCPSE_H

#include "polynomial.hpp"
#include "factorial.hpp"
#include "eigenlib.hpp"
#include "knearestneighbors.hpp"
#include "primitive.hpp"

namespace umuq
{

/*! \defgroup Numerics_Module Numerics module
 * This is the numerics module of %UMUQ providing all necessary classes of numerical computation.
 */

/*! 
 * \defgroup DCPSE Discretization-Corrected PSE Operators
 * \ingroup Numerics_Module 
 */

/*! \class dcpse
 * \ingroup DCPSE
 * 
 * \brief General class for (DC-PSE) \link DCPSE Discretization-Corrected PSE Operators \endlink
 * 
 * It creates a discretized differential operator and interpolators \ref 
 * 
 * \tparam T         Data type
 * \tparam Distance  Distance type for computing the distances to the nearest neighbors
 *                   (Default is a specialized class - \b kNearestNeighbor<T> with L2 distance)<br>
 *                   \sa umuq::kNearestNeighbor.<br>
 *                   \sa umuq::L2NearestNeighbor.<br>
 *                   \sa umuq::MahalanobisNearestNeighbor.
 */
/*!
 * \todo
 * Currently the class works only for one term and it should be extended to multi terms
 */
template <typename T, class Distance = L2NearestNeighbor<T>>
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
    explicit dcpse(dcpse<T, Distance> &&other);

    /*!
     * \brief Move assignment operator
     * 
     * \param other dcpse object
     * 
     * \returns dcpse<T, Distance>& dcpse object
     */
    dcpse<T, Distance> &operator=(dcpse<T, Distance> &&other);

    /*!
     * \brief Destroy the dcpse object
     * 
     */
    ~dcpse() {}

    /*!
     * \brief Computes generalized DC-PSE differential operators on set of input points
     *
     * This function uses one set of points as input data to compute the generalized DC-PSE 
     * differential operators.<br>
     * If the degree of the differential operator is zero \f$ | \beta | = 0 \f$, suggests one 
     * should use the interpolator function not this one. 
     * 
     * \param idata    A pointer to input data
     * \param nPoints  Number of data points
     * \param beta     In multi-dimensional notation \f$ \beta=\left(\beta_1, \cdots, \beta_d \right). \f$<br>
     *                 Notation for partial derivatives:<br>
     *                 \f$  D^\beta = \frac{\partial^{|\beta|}}{\partial x_1^{\beta_1} \partial x_2^{\beta_2}\cdots\partial x_d^{\beta_d}}. \f$
     * \param order    Order of accuracy (default is 2nd order accurate)
     * \param nENN     Number of extra nearest neighbors to aid in case of singularity of the Vandermonde matrix (default is 2)
     * \param ratio    The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     * 
     */
    bool computeWeights(T *idata, int const nPoints, int *beta, int order = 2, int nENN = 2, T ratio = static_cast<T>(1));

    /*!
     * \brief Computes generalized DC-PSE differential operators on the set of query points.
     * 
     * This function uses one set of points as input data to compute the generalized DC-PSE 
     * differential opearators on the set of query points.<br>
     * If the degree of the differential operator is zero \f$ | \beta | = 0 \f$, means one should
     * use the interpolator function not this one. 
     * 
     * \param idata     A pointer to input data
     * \param nPoints   Number of data points
     * \param qdata     A pointer to query data
     * \param nqPoints  Number of query data points
     * \param beta      In multi-dimensional notation \f$ \beta=\left(\beta_1, \cdots, \beta_d \right). \f$<br>
     *                  Notation for partial derivatives:<br>
     *                  \f$ D^\beta = \frac{\partial^{|\beta|}} {\partial x_1^{\beta_1} \partial x_2^{\beta_2}\cdots\partial x_d^{\beta_d}}. \f$
     * \param order     Order of accuracy (default is 2nd order accurate)
     * \param nENN      Number of extra nearest neighbors to aid in case of singularity of the Vandermonde matrix (default is 2)
     * \param ratio     The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     */
    bool computeWeights(T *idata, int const nPoints, T *qdata, int const nqPoints, int *beta, int order = 2, int nENN = 2, T ratio = static_cast<T>(1));

    /*! \fn computeInterpolatorWeights
     * \brief Computes generalized DC-PSE interpolator operators on the set of points.
     * 
     * This function uses one set of points as input data to compute the generalized DC-PSE 
     * interpolator operators on them.
     * 
     * \param idata    A pointer to input data 
     * \param nPoints  Number of data points
     * \param order    Order of accuracy (default is 2nd order accurate)
     * \param nENN     Number of extra nearest neighbors to aid in case of singularity of the Vandermonde matrix (default is 2)
     * \param ratio    The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     */
    bool computeInterpolatorWeights(T *idata, int const nPoints, int order = 2, int nENN = 2, T ratio = static_cast<T>(1));

    /*!
     * \brief Computes generalized DC-PSE interpolator operators on the set of query points.
     * 
     * This function uses one set of points as input data to compute the generalized DC-PSE 
     * interpolator operators on the set of query points.
     * 
     * \param idata     A pointer to input data 
     * \param nPoints   Number of data points
     * \param qdata     A pointer to query data 
     * \param nqPoints  Number of query data points
     * \param order     Order of accuracy (default is 2nd order accurate)
     * \param nENN      Number of extra nearest neighbors to aid in case of singularity of the Vandermonde matrix (default is 2)
     * \param ratio     The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     */
    bool computeInterpolatorWeights(T *idata, int const nPoints, T *qdata, int const nqPoints, int order = 2, int nENN = 2, T ratio = static_cast<T>(1));

    /*!
     * \brief Evaluate a discretized DC-PSE operator from function values of input data and put the results as the query data function values
     * 
     * This function uses function values of input data and the weights of the operator which have 
     * been previously computed to compute the query values and put the results as the query data 
     * function values. <br>
     * At first it checks the computed kernel size to be equal to the number of query points times the 
     * size of monomials which has been previously computed for the required degree of the DC-PSE operator.
     * 
     * \param iFvalue   A pointer to input data function value
     * \param nPoints   Number of data points
     * \param qFvalue   A pointer to query data function value
     * \param nqPoints  Number of query data points
     */
    bool compute(T *iFvalue, int const nPoints, T *qFvalue, int const nqPoints);

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
     * \param iFvalue   A pointer to input data function value
     * \param nPoints   Number of data points
     * \param qFvalue   A pointer to query data function value
     * \param nqPoints  Number of query data points
     */
    bool interpolate(T *iFvalue, int const nPoints, T *&qFvalue, int const nqPoints);

    /*!
     * \brief A pointer to neighborhood kernel at index
     * 
     * \param index Index of a point (from query data points) to get its neighborhood kernel
     * 
     * \returns A (pointer to a) row of the nearest neighbors kernel values.
     */
    inline T *neighborhoodKernel(int const index) const;

    /*!
     * \brief A pointer to kernel array of all query points
     * 
     * \returns A pointer to kernel array of all query points
     */
    inline T *neighborhoodKernel() const;

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
    inline T averageSpace(int const index) const;

    /*!
     * \brief A pointer to component-wise average neighbor spacing
     * 
     * \returns A pointer to component-wise average neighbor spacing
     */
    inline T *averageSpace() const;

  private:
    /*!
     * \brief Delete a dcpse object copy construction
     * 
     * Make it noncopyable.
     */
    dcpse(dcpse<T, Distance> const &) = delete;

    /*!
     * \brief Delete a dcpse object assignment
     * 
     * Make it nonassignable
     * 
     * \returns dcpse<T, Distance>& 
     */
    dcpse<T, Distance> &operator=(dcpse<T, Distance> const &) = delete;

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
    std::vector<T> dcKernel;

    //! k-NearestNeighbor Object
    std::unique_ptr<Distance> KNN;

    //! Component-wise average neighbor spacing \f$ h = \frac{1}{N} \sum_{p=1}^{N}\left(|x_{1}-x_{p1}| + \cdots |x_{d} -x_{pd}| \right), \f$
    std::vector<T> h_average;

    //! The sign is chosen positive for odd \f$ | \beta | \f$ and negative for even \f$ | \beta | \f$
    T rhscoeff;
};

template <typename T, class Distance>
dcpse<T, Distance>::dcpse(int ndim, int nterms) : nDim(ndim),
                                                  nTerms(nterms),
                                                  dcMonomialSize(0),
                                                  dcKernelSize(0),
                                                  Order(nterms) {}

template <typename T, class Distance>
dcpse<T, Distance>::dcpse(dcpse<T, Distance> &&other)
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

template <typename T, class Distance>
dcpse<T, Distance> &dcpse<T, Distance>::operator=(dcpse<T, Distance> &&other)
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

template <typename T, class Distance>
bool dcpse<T, Distance>::computeWeights(T *idata, int const nPoints, int *beta, int order, int nENN, T ratio)
{
    if (nPoints < 1)
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
    ratio = (ratio > 0) ? ratio : static_cast<T>(1);

    // \f$ |\beta| = \beta_1 + \cdots + \beta_d \f$
    int Beta = std::accumulate(beta, beta + nDim, 0);
    if (Beta == 0)
    {
        UMUQWARNING("Zero order degree derivative gives an approximation! \n If this is an interpolation use the interpolation function!");
    }

    int alphamin = (Beta % 2 == 0);

    // \f$ (-1)^{|\beta|} \f$
    rhscoeff = alphamin ? static_cast<T>(1) : -static_cast<T>(1);

    // Create an instance of polynomial object with polynomial degree of \f$ |\beta| + r -1 \f$
    polynomial<T> poly(nDim, order + Beta - 1);

    /*
     * Get the monomials size
     * \f$ \text{monomialSize} = \left(\begin{matrix} |\beta| + r + d -1 \\ d \end{matrix}\right) - \alpha_{\min} \f$
     */
    dcMonomialSize = poly.monomialsize() - alphamin;

    if (nPoints * dcMonomialSize > dcKernelSize)
    {
        dcKernelSize = nPoints * dcMonomialSize;
        try
        {
            // Make sure of the correct kernel size
            dcKernel.resize(dcKernelSize);

            h_average.resize(nPoints);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        dcKernelSize = nPoints * dcMonomialSize;
    }

    if (KNN)
    {
        if (nPoints != KNN->numInputdata() || nPoints != KNN->numQuerydata())
        {
            try
            {
                /*
                 * Finding K nearest neighbors
                 * The number of points K in the neighborhood of each point
                 * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
                 */
                KNN.reset(new Distance(nPoints, nDim, dcMonomialSize + nENN));
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
            KNN.reset(new Distance(nPoints, nDim, dcMonomialSize + nENN));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    // Construct a kd-tree index & do nearest neighbors search
    KNN->buildIndex(idata);

    /*
     * Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
     * \f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b}  \f$
     */
    EVectorX<T> RHSB(dcMonomialSize);
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
                RHSB(i) = T{};
            }
            else
            {
                T fact = static_cast<T>(1);
                std::for_each(beta, beta + nDim, [&](int const b_j) { fact *= factorial<T>(b_j); });
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
        if (rhscoeff > T{})
        {
            std::ptrdiff_t const id = alphamin * nDim;
            if (std::accumulate(alpha + id, alpha + id + nDim, 0) == 0)
            {
                RHSB(0) = static_cast<T>(5);
            }
        }
    }

    // Total number of nearest neighbors for each point
    int nNN = KNN->numNearestNeighbors();

    /*
     * Creating a transpose of the Vandermonde matrix
     * with the size of monomials * monomials \f$  = l \times l \f$
     */
    EMatrixX<T> VMT(dcMonomialSize, dcMonomialSize);
    EMatrixX<T> VMTimage(dcMonomialSize, nNN);

    // Matrix of exponential window function
    EVectorX<T> EM(dcMonomialSize);
    EVectorX<T> EMimage(nNN);

    // Matrix A of a linear system for the kernel coefficients
    EMatrixX<T> AM(dcMonomialSize, dcMonomialSize);

    // Matrix B
    EMatrixX<T> BMT(dcMonomialSize, dcMonomialSize);
    EMatrixX<T> BMTimage(dcMonomialSize, nNN);

    // ${\mathbf a}^T({\mathbf x})$ is the column vector of coefficients which is the solution of linear system
    EVectorX<T> SV(dcMonomialSize);

    // Array for keeping the component-wise L1 distances
    std::vector<T> L1Dist(nNN * nDim);

    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
    T *column = new T[dcMonomialSize + alphamin];

    std::vector<int> IndexId(dcMonomialSize);

    // Number of points with singular Vandermonde matrix
    int nPointsWithSingularVandermondeMatrix(0);

    // Loop over all points
    for (int i = 0; i < nPoints; i++)
    {
        std::ptrdiff_t const IdM = i * dcMonomialSize;
        std::ptrdiff_t const IdI = i * nDim;

        // A pointer to nearest neighbors indices of point i
        int *NearestNeighbors = KNN->NearestNeighbors(i);

        // A pointer to nearest neighbors square distances from the point i
        T *nnDist = KNN->NearestNeighborsDistances(i);

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
            T *Idata = idata + IdI;

            // \f$ $\left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x} - {\mathbf x}_p \right\} \f$
            for (int j = 0, n = 0; j < nNN; j++)
            {
                std::ptrdiff_t const IdJ = NearestNeighbors[j] * nDim;

                // pointer to idata (neighbors of i)
                T *Jdata = idata + IdJ;

                for (int d = 0; d < nDim; d++, n++)
                {
                    L1Dist[n] = Idata[d] - Jdata[d];
                }
            }
        }

        // Compute component-wise average neighbor spacing
        T h_avg(0);
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](T const l_i) { h_avg += std::abs(l_i); });

        // Component-wise average neighbor spacing \f$ h \f$
        h_avg /= static_cast<T>(nNN);

        h_average[i] = h_avg;

        // Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
        T const byEpsilon = ratio / h_avg;
        T const byEpsilonsq = byEpsilon * byEpsilon;
        T const byEpsilonsq2 = 0.5 * byEpsilonsq;
        T const byEpsilonPowerBeta = std::pow(byEpsilon, Beta);

        // Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](T &l_i) { l_i *= byEpsilon; });

        // Loop through the neighbors
        for (int j = 0; j < dcMonomialSize; j++)
        {
            // Id in the L1 distance list
            std::ptrdiff_t const Id = j * nDim;

            // Evaluates a monomial at a point \f$ {\mathbf x} \f$
            poly.monomialValue(L1Dist.data() + Id, column);

            EVectorMapType<T> columnV(column + alphamin, dcMonomialSize);

            // Fill the Vandermonde matrix column by column
            VMT.block(0, j, dcMonomialSize, 1) << columnV;
        }

        for (int j = 0; j < dcMonomialSize; j++)
        {
            EM(j) = std::exp(-nnDist[j] * byEpsilonsq2);
        }

        int dcrank;

        {
            // LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<T>> lu(VMT);

            dcrank = lu.rank();

            if (dcrank < dcMonomialSize && dcrank >= dcMonomialSize - nENN)
            {
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    IndexId[j] = lu.permutationQ().indices()(j);
                }
            }
        }

        if (dcrank < dcMonomialSize)
        {
            // We have a singular Vandermonde matrix
            nPointsWithSingularVandermondeMatrix++;

            // If necessary, remove redundant equations/coefficients

            // Number of neighbor points are not enough
            if (dcrank < dcMonomialSize - nENN)
            {
                UMUQWARNING("Number of neighbor points are not enough! Matrix rank =");
                std::cerr << dcrank << " < " << dcMonomialSize - nENN << std::endl;

                if (nENN > 0)
                {
                    VMTimage.block(0, 0, dcMonomialSize, dcMonomialSize) << VMT;
                    EMimage.head(dcMonomialSize) << EM;

                    // Loop through the rest of nearest neighbors
                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        // Id in the list
                        std::ptrdiff_t const Id = j * nDim;

                        // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                        poly.monomialValue(L1Dist.data() + Id, column);

                        EVectorMapType<T> columnV(column + alphamin, dcMonomialSize);

                        // Fill the Vandermonde matrix column by column
                        VMTimage.block(0, j, dcMonomialSize, 1) << columnV;
                    }

                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        EMimage(j) = std::exp(-nnDist[j] * byEpsilonsq2);
                    }

                    /* 
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMTimage = VMTimage * EMatrixX<T>(EMimage.asDiagonal());
                    AM = BMTimage * BMTimage.transpose();
                }
                else
                {
                    /* 
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMT = VMT * EMatrixX<T>(EM.asDiagonal());
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
                for (int j = dcrank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t const Id = k * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<T> columnV(column + alphamin, dcMonomialSize);

                    // Fill the Vandermonde matrix by the new column
                    VMT.block(0, l, dcMonomialSize, 1) << columnV;
                }

                for (int j = dcrank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    EM(l) = std::exp(-nnDist[k] * byEpsilonsq2);
                }

                /*
                 * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                 *  {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                 *  {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                 * \f}
                 */
                BMT = VMT * EMatrixX<T>(EM.asDiagonal());
                AM = BMT * BMT.transpose();
            }

            {
                /*
                 * Two-sided Jacobi SVD decomposition, ensuring optimal reliability and accuracy.
                 * Thin U and V are all we need for (least squares) solving.
                 */
                Eigen::JacobiSVD<EMatrixX<T>> svd(AM, Eigen::DecompositionOptions::ComputeThinU | Eigen::DecompositionOptions::ComputeThinV);

                /*
                 * SV contains the least-squares solution of 
                 * \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b} \f$
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
            if (dcrank < dcMonomialSize - nENN)
            {
                // Loop through the neighbors
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    // Id in the list
                    std::ptrdiff_t const Id = j * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<T> columnV(column + alphamin, dcMonomialSize);

                    T const expo = std::exp(-nnDist[j] * byEpsilonsq);

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
                    T expo;

                    if (j >= dcrank)
                    {
                        // Id in the list
                        Id = m * nDim;
                        expo = std::exp(-nnDist[m] * byEpsilonsq);
                        m++;
                    }
                    else
                    {
                        Id = l * nDim;
                        expo = std::exp(-nnDist[l] * byEpsilonsq);
                    }

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<T> columnV(column + alphamin, dcMonomialSize);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + l;
                    dcKernel[IdK] = SV.dot(columnV) * byEpsilonPowerBeta * expo;
                }

                // Loop through the neighbors
                for (int j = dcrank, m = dcMonomialSize; j < dcMonomialSize; j++, m++)
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
             * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
             * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
             * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
             * \f}
             */
            BMT = VMT * EMatrixX<T>(EM.asDiagonal());

            AM = BMT * BMT.transpose();

            // SV contains the solution of \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b} \f$
            SV = AM.lu().solve(RHSB);

            // Loop through the neighbors
            for (int j = 0; j < dcMonomialSize; j++)
            {
                // Id in the list
                std::ptrdiff_t const Id = j * nDim;

                // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                poly.monomialValue(L1Dist.data() + Id, column);

                EVectorMapType<T> columnV(column + alphamin, dcMonomialSize);

                T const expo = std::exp(-nnDist[j] * byEpsilonsq);

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

template <typename T, class Distance>
bool dcpse<T, Distance>::computeWeights(T *idata, int const nPoints, T *qdata, int const nqPoints, int *beta, int order, int nENN, T ratio)
{
    if (nPoints < 1)
    {
        UMUQFAILRETURN("Number of input data points are negative!");
    }

    if (nqPoints < 1)
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
    ratio = (ratio > 0) ? ratio : static_cast<T>(1);

    // \f$ |\beta| = \beta_1 + \cdots + \beta_d \f$
    int Beta = std::accumulate(beta, beta + nDim, 0);
    if (Beta == 0)
    {
        UMUQWARNING("Zero order degree derivative gives an approximation! \n If this is an interpolation use the interpolation function!");
    }

    int alphamin = (Beta % 2 == 0);

    // \f$ (-1)^{|\beta|} \f$
    rhscoeff = alphamin ? static_cast<T>(1) : -static_cast<T>(1);

    // Create an instance of polynomial object with polynomial degree of \f$ |\beta| + r -1 \f$
    polynomial<T> poly(nDim, order + Beta - 1);

    /*
     * Get the monomials size
     * \f$ \text{monomialSize} = \left(\begin{matrix} |\beta| + r + d -1 \\ d \end{matrix}\right) - \alpha_{\min} \f$
     */
    dcMonomialSize = poly.monomialsize() - alphamin;

    if (nqPoints * dcMonomialSize > dcKernelSize)
    {
        dcKernelSize = nqPoints * dcMonomialSize;
        try
        {
            // Make sure of the correct kernel size
            dcKernel.resize(dcKernelSize);

            h_average.resize(nqPoints);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        dcKernelSize = nqPoints * dcMonomialSize;
    }

    if (KNN)
    {
        if (nqPoints != KNN->numInputdata() || nqPoints != KNN->numQuerydata())
        {
            try
            {
                /*
                 * Finding K nearest neighbors
                 * The number of points K in the neighborhood of each point
                 * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
                 */
                KNN.reset(new Distance(nPoints, nqPoints, nDim, dcMonomialSize + nENN));
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
            KNN.reset(new Distance(nPoints, nqPoints, nDim, dcMonomialSize + nENN));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    // Construct a kd-tree index & do nearest neighbors search
    KNN->buildIndex(idata, qdata);

    /*
     * Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
     * \f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b}  \f$
     */
    EVectorX<T> RHSB(dcMonomialSize);
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
                RHSB(i) = T{};
            }
            else
            {
                T fact = static_cast<T>(1);
                std::for_each(beta, beta + nDim, [&](int const b_j) { fact *= factorial<T>(b_j); });
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
        RHSB(0) = T{};
    }

    // Total number of nearest neighbors for each point
    int nNN = KNN->numNearestNeighbors();

    // Array for keeping the component-wise L1 distances
    std::vector<T> L1Dist(nNN * nDim);

    /*
     * Creating a transpose of the Vandermonde matrix
     * with the size of monomials * monomials \f$  = l \times l \f$
     */
    EMatrixX<T> VMT(dcMonomialSize, dcMonomialSize);
    EMatrixX<T> VMTimage(dcMonomialSize, nNN);

    // Matrix of exponential window function
    EVectorX<T> EM(dcMonomialSize);
    EVectorX<T> EMimage(nNN);

    // Matrix A of a linear system for the kernel coefficients
    EMatrixX<T> AM(dcMonomialSize, dcMonomialSize);

    // Matrix B
    EMatrixX<T> BMT(dcMonomialSize, dcMonomialSize);
    EMatrixX<T> BMTimage(dcMonomialSize, nNN);

    // ${\mathbf a}^T({\mathbf x})$ is the column vector of coefficients which is the solution of linear system
    EVectorX<T> SV(dcMonomialSize);

    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
    T *column = new T[dcMonomialSize + alphamin];

    std::vector<int> IndexId(dcMonomialSize);

    // Number of points with singular Vandermonde matrix
    int nPointsWithSingularVandermondeMatrix(0);

    // Loop over all query points
    for (int i = 0; i < nqPoints; i++)
    {
        std::ptrdiff_t const IdM = i * dcMonomialSize;
        std::ptrdiff_t const IdI = i * nDim;

        // A pointer to nearest neighbors indices of point i
        int *NearestNeighbors = KNN->NearestNeighbors(i);

        // A pointer to nearest neighbors square distances from the point i
        T *nnDist = KNN->NearestNeighborsDistances(i);

        /*
         * For each point \f$ {\mathbf x} \f$ we define \f$ \left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x}_p - {\mathbf x} \right\}, \f$
         * as the set of vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points \f${\mathbf x}_p\f$ in the support of \f${\mathbf x}\f$.
         */

        {
            // pointer to query data
            T *Idata = qdata + IdI;

            // \f$ $\left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x} - {\mathbf x}_p \right\} \f$
            for (int j = 0, n = 0; j < nNN; j++)
            {
                std::ptrdiff_t const IdJ = NearestNeighbors[j] * nDim;

                // pointer to idata (neighbors of i)
                T *Jdata = idata + IdJ;

                for (int d = 0; d < nDim; d++, n++)
                {
                    L1Dist[n] = Idata[d] - Jdata[d];
                }
            }
        }

        // Compute component-wise average neighbor spacing
        T h_avg(0);
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](T const l_i) { h_avg += std::abs(l_i); });

        // Component-wise average neighbor spacing \f$ h \f$
        h_avg /= static_cast<T>(nNN);

        h_average[i] = h_avg;

        // Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
        T const byEpsilon = ratio / h_avg;
        T const byEpsilonsq = byEpsilon * byEpsilon;
        T const byEpsilonsq2 = 0.5 * byEpsilonsq;
        T const byEpsilonPowerBeta = std::pow(byEpsilon, Beta);

        // Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](T &l_i) { l_i *= byEpsilon; });

        // Loop through the neighbors
        for (int j = 0; j < dcMonomialSize; j++)
        {
            // Id in the L1 distance list
            std::ptrdiff_t const Id = j * nDim;

            // Evaluates a monomial at a point \f$ {\mathbf x} \f$
            poly.monomialValue(L1Dist.data() + Id, column);

            EVectorMapType<T> columnV(column + alphamin, dcMonomialSize);

            // Fill the Vandermonde matrix column by column
            VMT.block(0, j, dcMonomialSize, 1) << columnV;
        }

        for (int j = 0; j < dcMonomialSize; j++)
        {
            EM(j) = std::exp(-nnDist[j] * byEpsilonsq2);
        }

        int dcrank;

        {
            // LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<T>> lu(VMT);

            dcrank = lu.rank();

            if (dcrank < dcMonomialSize && dcrank >= dcMonomialSize - nENN)
            {
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    IndexId[j] = lu.permutationQ().indices()(j);
                }
            }
        }

        if (dcrank < dcMonomialSize)
        {
            // We have a singular Vandermonde matrix
            nPointsWithSingularVandermondeMatrix++;

            // if necessary, remove redundant equations/coefficients

            // Number of neighbor points are not enough
            if (dcrank < dcMonomialSize - nENN)
            {
                UMUQWARNING("Number of neighbor points are not enough! Matrix rank = ");
                std::cerr << dcrank << " < " << dcMonomialSize - nENN << std::endl;

                if (nENN > 0)
                {
                    VMTimage.block(0, 0, dcMonomialSize, dcMonomialSize) << VMT;
                    EMimage.head(dcMonomialSize) << EM;

                    // Loop through the rest of nearest neighbors
                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        // Id in the list
                        std::ptrdiff_t const Id = j * nDim;

                        // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                        poly.monomialValue(L1Dist.data() + Id, column);

                        EVectorMapType<T> columnV(column + alphamin, dcMonomialSize);

                        // Fill the Vandermonde matrix column by column
                        VMTimage.block(0, j, dcMonomialSize, 1) << columnV;
                    }

                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        EMimage(j) = std::exp(-nnDist[j] * byEpsilonsq2);
                    }

                    /*
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMTimage = VMTimage * EMatrixX<T>(EMimage.asDiagonal());
                    AM = BMTimage * BMTimage.transpose();
                }
                else
                {
                    /* 
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMT = VMT * EMatrixX<T>(EM.asDiagonal());
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
                for (int j = dcrank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t const Id = k * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<T> columnV(column + alphamin, dcMonomialSize);

                    // Fill the Vandermonde matrix by the new column
                    VMT.block(0, l, dcMonomialSize, 1) << columnV;
                }

                for (int j = dcrank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    EM(l) = std::exp(-nnDist[k] * byEpsilonsq2);
                }

                /* 
                 * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                 * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                 * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                 * \f}
                 */
                BMT = VMT * EMatrixX<T>(EM.asDiagonal());
                AM = BMT * BMT.transpose();
            }

            {
                /*
                 * Two-sided Jacobi SVD decomposition, ensuring optimal reliability and accuracy.
                 * Thin U and V are all we need for (least squares) solving.
                 */
                Eigen::JacobiSVD<EMatrixX<T>> svd(AM, Eigen::DecompositionOptions::ComputeThinU | Eigen::DecompositionOptions::ComputeThinV);

                /*
                 * SV contains the least-squares solution of 
                 * \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b} \f$
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

            if (dcrank < dcMonomialSize - nENN)
            {
                // Loop through the neighbors
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    // Id in the list
                    std::ptrdiff_t const Id = j * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<T> columnV(column + alphamin, dcMonomialSize);

                    T const expo = std::exp(-nnDist[j] * byEpsilonsq);

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
                    T expo;

                    if (j >= dcrank)
                    {
                        // Id in the list
                        Id = m * nDim;
                        expo = std::exp(-nnDist[m] * byEpsilonsq);
                        m++;
                    }
                    else
                    {
                        Id = l * nDim;
                        expo = std::exp(-nnDist[l] * byEpsilonsq);
                    }

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<T> columnV(column + alphamin, dcMonomialSize);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + l;
                    dcKernel[IdK] = SV.dot(columnV) * byEpsilonPowerBeta * expo;
                }

                // Loop through the neighbors
                for (int j = dcrank, m = dcMonomialSize; j < dcMonomialSize; j++, m++)
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
             * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
             * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
             * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
             * \f}
             */
            BMT = VMT * EMatrixX<T>(EM.asDiagonal());

            AM = BMT * BMT.transpose();

            // SV contains the solution of \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b} \f$
            SV = AM.lu().solve(RHSB);

            // Loop through the neighbors
            for (int j = 0; j < dcMonomialSize; j++)
            {
                // Id in the list
                std::ptrdiff_t const Id = j * nDim;

                // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                poly.monomialValue(L1Dist.data() + Id, column);

                EVectorMapType<T> columnV(column + alphamin, dcMonomialSize);

                T const expo = std::exp(-nnDist[j] * byEpsilonsq);

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

template <typename T, class Distance>
bool dcpse<T, Distance>::computeInterpolatorWeights(T *idata, int const nPoints, int order, int nENN, T ratio)
{
    if (nPoints < 1)
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
    ratio = (ratio > 0) ? ratio : static_cast<T>(1);

    // Create an instance of a polynomial object with polynomial degree of \f$ |\beta| + r - 1 \f$
    polynomial<T> poly(nDim, order - 1);

    /* 
     * Get the monomials size
     * \f$ \text{monomialSize} = \left(\begin{matrix} r + d -1 \\ d \end{matrix}\right) \f$
     */
    dcMonomialSize = poly.monomialsize();

    if (nPoints * dcMonomialSize > dcKernelSize)
    {
        dcKernelSize = nPoints * dcMonomialSize;
        try
        {
            // Make sure of the correct kernel size
            dcKernel.resize(dcKernelSize);

            h_average.resize(nPoints);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        dcKernelSize = nPoints * dcMonomialSize;
    }

    if (KNN)
    {
        if (nPoints != KNN->numInputdata() || nPoints != KNN->numQuerydata())
        {
            try
            {
                /* 
                 * Finding K nearest neighbors
                 * The number of points K in the neighborhood of each point
                 * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
                 */
                KNN.reset(new Distance(nPoints, nPoints, nDim, dcMonomialSize + nENN));
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
            KNN.reset(new Distance(nPoints, nPoints, nDim, dcMonomialSize + nENN));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    // Construct a kd-tree index & do nearest neighbors search
    KNN->buildIndex(idata);

    /*
     * Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
     * \f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b}  \f$
     */
    EVectorX<T> RHSB0 = EVectorX<T>::Zero(dcMonomialSize);
    RHSB0(0) = static_cast<T>(1);
    EVectorX<T> RHSB(dcMonomialSize);

    // Total number of nearest neighbours for each point
    int nNN = KNN->numNearestNeighbors();

    /*
     * Creating a transpose of the Vandermonde matrix
     * with the size of monomials * monomials \f$  = l \times l \f$
     */
    EMatrixX<T> VMT(dcMonomialSize, dcMonomialSize);
    EMatrixX<T> VMTimage(dcMonomialSize, nNN);

    // Matrix of exponential window function
    EVectorX<T> EM(dcMonomialSize);
    EVectorX<T> EMimage(nNN);

    EVectorX<T> columnL(dcMonomialSize);

    // Matrix A of a linear system for the kernel coefficients
    EMatrixX<T> AM(dcMonomialSize, dcMonomialSize);

    // Matrix B^T
    EMatrixX<T> BMT(dcMonomialSize, dcMonomialSize);
    EMatrixX<T> BMTimage(dcMonomialSize, nNN);

    // ${\mathbf a}^T({\mathbf x})$ is the column vector of coefficients which is the solution of linear system
    EVectorX<T> SV(dcMonomialSize);

    // Array for keeping the component-wise L1 distances
    std::vector<T> L1Dist(nNN * nDim);

    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
    T *column = new T[dcMonomialSize];

    // Array to kepp indexing
    std::vector<int> IndexId(dcMonomialSize);

    // Number of points with singular Vandermonde matrix
    int nPointsWithSingularVandermondeMatrix(0);

    // Loop over all query points
    for (int i = 0; i < nPoints; i++)
    {
        // Index inside kernel
        std::ptrdiff_t const IdM = i * dcMonomialSize;

        // Index in idata array
        std::ptrdiff_t const IdI = i * nDim;

        // A pointer to nearest neighbors indices of point i
        int *NearestNeighbors = KNN->NearestNeighbors(i);

        // A pointer to nearest neighbors square distances from the point i
        T *nnDist = KNN->NearestNeighborsDistances(i);

        /*
         * For each point \f$ {\mathbf x} \f$ we define 
         * \f$ \left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x}_p - {\mathbf x} \right\}, \f$
         * as the set of vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points 
         * \f${\mathbf x}_p\f$ in the support of \f${\mathbf x}\f$.
         */
        {
            // pointer to query data
            T *Idata = idata + IdI;

            // \f$ $\left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x} - {\mathbf x}_p \right\} \f$
            for (int j = 0, n = 0; j < nNN; j++)
            {
                // Neighbor index in idata array
                std::ptrdiff_t const IdJ = NearestNeighbors[j] * nDim;

                // pointer to idata (neighbors of i)
                T *Jdata = idata + IdJ;

                for (int d = 0; d < nDim; d++, n++)
                {
                    L1Dist[n] = Idata[d] - Jdata[d];
                }
            }
        }

        // Compute component-wise average neighbor spacing
        T h_avg(0);
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](T const l_i) { h_avg += std::abs(l_i); });

        // Component-wise average neighbor spacing \f$ h \f$
        h_avg /= static_cast<T>(nNN);

        h_average[i] = h_avg;

        // Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
        T const byEpsilon = ratio / h_avg;
        T const byEpsilonsq = byEpsilon * byEpsilon;
        T const byEpsilonsq2 = 0.5 * byEpsilonsq;

        // Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](T &l_i) { l_i *= byEpsilon; });

        // Use the correct RHS for each point
        RHSB = RHSB0;

        // Loop through the neighbors
        for (int j = 0; j < dcMonomialSize; j++)
        {
            // Id in the L1 distance list
            std::ptrdiff_t const Id = j * nDim;

            // Evaluates a monomial at a point \f$ {\mathbf x} \f$
            poly.monomialValue(L1Dist.data() + Id, column);

            EVectorMapType<T> columnV(column, dcMonomialSize);

            // Fill the Vandermonde matrix column by column
            VMT.block(0, j, dcMonomialSize, 1) << columnV;
        }

        for (int j = 0; j < dcMonomialSize; j++)
        {
            EM(j) = std::exp(-nnDist[j] * byEpsilonsq2);
        }

        int dcrank;

        {
            // LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<T>> lu(VMT);

            dcrank = lu.rank();

            if (dcrank < dcMonomialSize && dcrank >= dcMonomialSize - nENN)
            {
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    IndexId[j] = lu.permutationQ().indices()(j);
                }
            }
        }

        if (dcrank < dcMonomialSize)
        {
            // We have a singular Vandermonde matrix
            nPointsWithSingularVandermondeMatrix++;

            // if necessary, remove redundant equations/coefficients

            // Number of neighbor points are not enough
            if (dcrank < dcMonomialSize - nENN)
            {
                UMUQWARNING("Number of neighbor points are not enough! Matrix rank = ");
                std::cerr << dcrank << " < " << dcMonomialSize - nENN << std::endl;

                if (nENN > 0)
                {
                    VMTimage.block(0, 0, dcMonomialSize, dcMonomialSize) << VMT;
                    EMimage.head(dcMonomialSize) << EM;

                    // Loop through the rest of nearest neighbors
                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        // Id in the list
                        std::ptrdiff_t const Id = j * nDim;

                        // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                        poly.monomialValue(L1Dist.data() + Id, column);

                        EVectorMapType<T> columnV(column, dcMonomialSize);

                        // Fill the Vandermonde matrix column by column
                        VMTimage.block(0, j, dcMonomialSize, 1) << columnV;
                    }

                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        EMimage(j) = std::exp(-nnDist[j] * byEpsilonsq2);
                    }

                    /* 
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMTimage = VMTimage * EMatrixX<T>(EMimage.asDiagonal());
                    AM = BMTimage * BMTimage.transpose();
                }
                else
                {
                    /* 
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMT = VMT * EMatrixX<T>(EM.asDiagonal());
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
                for (int j = dcrank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t const Id = k * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<T> columnV(column, dcMonomialSize);

                    // Get the column l which causes singularity
                    columnL << VMT.block(0, l, dcMonomialSize, 1);

                    // Fill the Vandermonde matrix by the new column
                    VMT.block(0, l, dcMonomialSize, 1) << columnV;
                }

                for (int j = dcrank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    EM(l) = std::exp(-nnDist[k] * byEpsilonsq2);
                }

                /* 
                 * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                 * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                 * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                 * \f}
                 */
                BMT = VMT * EMatrixX<T>(EM.asDiagonal());
                AM = BMT * BMT.transpose();
            }

            {
                /*
                 * Two-sided Jacobi SVD decomposition, ensuring optimal reliability and accuracy.
                 * Thin U and V are all we need for (least squares) solving.
                 */
                Eigen::JacobiSVD<EMatrixX<T>> svd(AM, Eigen::DecompositionOptions::ComputeThinU | Eigen::DecompositionOptions::ComputeThinV);

                /*
                 * SV contains the least-squares solution of 
                 * \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b} \f$
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

            if (dcrank < dcMonomialSize - nENN)
            {
                // Loop through the neighbors
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    // Id in the list
                    std::ptrdiff_t const Id = j * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<T> columnV(column, dcMonomialSize);

                    T const expo = std::exp(-nnDist[j] * byEpsilonsq);

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
                    T expo;

                    if (j >= dcrank)
                    {
                        // Id in the list
                        Id = m * nDim;
                        expo = std::exp(-nnDist[m] * byEpsilonsq);
                        m++;
                    }
                    else
                    {
                        Id = l * nDim;
                        expo = std::exp(-nnDist[l] * byEpsilonsq);
                    }

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<T> columnV(column, dcMonomialSize);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + l;
                    dcKernel[IdK] = SV.dot(columnV) * expo;
                }

                // Loop through the neighbors
                for (int j = dcrank, m = dcMonomialSize; j < dcMonomialSize; j++, m++)
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
             * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
             * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
             * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
             * \f}
             */
            BMT = VMT * EMatrixX<T>(EM.asDiagonal());

            AM = BMT * BMT.transpose();

            // SV contains the solution of \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b} \f$
            SV = AM.lu().solve(RHSB);

            // Loop through the neighbors
            for (int j = 0; j < dcMonomialSize; j++)
            {
                // Id in the list
                std::ptrdiff_t const Id = j * nDim;

                // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                poly.monomialValue(L1Dist.data() + Id, column);

                EVectorMapType<T> columnV(column, dcMonomialSize);

                T const expo = std::exp(-nnDist[j] * byEpsilonsq);

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

template <typename T, class Distance>
bool dcpse<T, Distance>::computeInterpolatorWeights(T *idata, int const nPoints, T *qdata, int const nqPoints, int order, int nENN, T ratio)
{
    if (nPoints < 1)
    {
        UMUQFAILRETURN("Number of input data points are negative!");
    }

    if (nqPoints < 1)
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
    ratio = (ratio > 0) ? ratio : static_cast<T>(1);

    // Create an instance of a polynomial object with polynomial degree of \f$ |\beta| + r - 1 \f$
    polynomial<T> poly(nDim, order - 1);

    /* 
     * Get the monomials size
     * \f$ \text{monomialSize} = \left(\begin{matrix} r + d -1 \\ d \end{matrix}\right) \f$
     */
    dcMonomialSize = poly.monomialsize();

    if (nqPoints * dcMonomialSize > dcKernelSize)
    {
        dcKernelSize = nqPoints * dcMonomialSize;
        try
        {
            // Make sure of the correct kernel size
            dcKernel.resize(dcKernelSize);

            h_average.resize(nqPoints);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        dcKernelSize = nqPoints * dcMonomialSize;
    }

    if (KNN)
    {
        if (nqPoints != KNN->numInputdata() || nqPoints != KNN->numQuerydata())
        {
            try
            {
                /* 
                 * Finding K nearest neighbors
                 * The number of points K in the neighborhood of each point
                 * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
                 */
                KNN.reset(new Distance(nPoints, nqPoints, nDim, dcMonomialSize + nENN));
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
            KNN.reset(new Distance(nPoints, nqPoints, nDim, dcMonomialSize + nENN));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    // Construct a kd-tree index & do nearest neighbors search
    KNN->buildIndex(idata, qdata);

    // Vector of all idata points' distances from their closest nearest neighbor
    T *idataminDist = nullptr;

    {
        // Finding only one nearest neighbor for the input data points
        Distance KNN1(nPoints, nDim, 1);

        // Construct a kd-tree index & do nearest neighbors search
        KNN1.buildIndex(idata);

        idataminDist = KNN1.minDist();

        if (idataminDist == nullptr)
        {
            return false;
        }
    }

    /*
     * Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
     * \f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b}  \f$
     */
    EVectorX<T> RHSB0 = EVectorX<T>::Zero(dcMonomialSize);
    RHSB0(0) = static_cast<T>(1);
    EVectorX<T> RHSB(dcMonomialSize);

    // Total number of nearest neighbours for each point
    int nNN = KNN->numNearestNeighbors();

    /*
     * Creating a transpose of the Vandermonde matrix
     * with the size of monomials * monomials \f$  = l \times l \f$
     */
    EMatrixX<T> VMT(dcMonomialSize, dcMonomialSize);
    EMatrixX<T> VMTimage(dcMonomialSize, nNN);

    // Matrix of exponential window function
    EVectorX<T> EM(dcMonomialSize);
    EVectorX<T> EMimage(nNN);

    EVectorX<T> columnL(dcMonomialSize);

    // Matrix A of a linear system for the kernel coefficients
    EMatrixX<T> AM(dcMonomialSize, dcMonomialSize);

    // Matrix B^T
    EMatrixX<T> BMT(dcMonomialSize, dcMonomialSize);
    EMatrixX<T> BMTimage(dcMonomialSize, nNN);

    // ${\mathbf a}^T({\mathbf x})$ is the column vector of coefficients which is the solution of linear system
    EVectorX<T> SV(dcMonomialSize);

    // Array for keeping the component-wise L1 distances
    std::vector<T> L1Dist(nNN * nDim);

    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
    T *column = new T[dcMonomialSize];

    // Array to kepp indexing
    std::vector<int> IndexId(dcMonomialSize);

    // Primitive (quartic spline) object
    quartic_spline<T> q;

    // Number of points with singular Vandermonde matrix
    int nPointsWithSingularVandermondeMatrix(0);

    // Loop over all query points
    for (int i = 0; i < nqPoints; i++)
    {
        // Index inside kernel
        std::ptrdiff_t const IdM = i * dcMonomialSize;

        // Index in qdata array
        std::ptrdiff_t const IdI = i * nDim;

        // A pointer to nearest neighbors indices of point i
        int *NearestNeighbors = KNN->NearestNeighbors(i);

        // A pointer to nearest neighbors square distances from the point i
        T *nnDist = KNN->NearestNeighborsDistances(i);

        /*
         * For each point \f$ {\mathbf x} \f$ we define 
         * \f$ \left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x}_p - {\mathbf x} \right\}, \f$
         * as the set of vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points 
         * \f${\mathbf x}_p\f$ in the support of \f${\mathbf x}\f$.
         */
        {
            // pointer to query data
            T *Idata = qdata + IdI;

            // \f$ $\left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x} - {\mathbf x}_p \right\} \f$
            for (int j = 0, n = 0; j < nNN; j++)
            {
                // Neighbor index in idata array
                std::ptrdiff_t const IdJ = NearestNeighbors[j] * nDim;

                // pointer to idata (neighbors of i)
                T *Jdata = idata + IdJ;

                for (int d = 0; d < nDim; d++, n++)
                {
                    L1Dist[n] = Idata[d] - Jdata[d];
                }
            }
        }

        // Compute component-wise average neighbor spacing
        T h_avg(0);
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](T const l_i) { h_avg += std::abs(l_i); });

        // Component-wise average neighbor spacing \f$ h \f$
        h_avg /= static_cast<T>(nNN);

        h_average[i] = h_avg;

        // Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
        T const byEpsilon = ratio / h_avg;
        T const byEpsilonsq = byEpsilon * byEpsilon;
        T const byEpsilonsq2 = 0.5 * byEpsilonsq;

        // Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](T &l_i) { l_i *= byEpsilon; });

        // Use the correct RHS for each point
        RHSB = RHSB0;

        // Loop through the neighbors
        for (int j = 0; j < dcMonomialSize; j++)
        {
            // Id in the L1 distance list
            std::ptrdiff_t const Id = j * nDim;

            // Evaluates a monomial at a point \f$ {\mathbf x} \f$
            poly.monomialValue(L1Dist.data() + Id, column);

            EVectorMapType<T> columnV(column, dcMonomialSize);

            // Fill the Vandermonde matrix column by column
            VMT.block(0, j, dcMonomialSize, 1) << columnV;

            // Neighbor point number
            int const IdJ = NearestNeighbors[j];

            /* 
             * Using a smooth correction function that satisfies
             * \f$ {\mathbf F} \left(\frac{{\mathbf x}_p-{\mathbf x}_q}{c({\mathbf x}_q)} \right) =\delta_{pq} \f$
             * Choose \f$ c({\mathbf x}) \f$ such that it is smaller than the distance
             * between the point and its nearest neighbors
             */
            T s = std::sqrt(nnDist[j]) / (0.9 * std::sqrt(idataminDist[IdJ]));

            // Compute the kernel value at the point
            T dckernelV = q.f(&s);

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
            EM(j) = std::exp(-nnDist[j] * byEpsilonsq2);
        }

        int dcrank;

        {
            // LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<T>> lu(VMT);

            dcrank = lu.rank();

            if (dcrank < dcMonomialSize && dcrank >= dcMonomialSize - nENN)
            {
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    IndexId[j] = lu.permutationQ().indices()(j);
                }
            }
        }

        if (dcrank < dcMonomialSize)
        {
            // We have a singular Vandermonde matrix
            nPointsWithSingularVandermondeMatrix++;

            // if necessary, remove redundant equations/coefficients

            // Number of neighbor points are not enough
            if (dcrank < dcMonomialSize - nENN)
            {
                UMUQWARNING("Number of neighbor points are not enough! Matrix rank = ");
                std::cerr << dcrank << " < " << dcMonomialSize - nENN << std::endl;

                if (nENN > 0)
                {
                    VMTimage.block(0, 0, dcMonomialSize, dcMonomialSize) << VMT;
                    EMimage.head(dcMonomialSize) << EM;

                    // Loop through the rest of nearest neighbors
                    for (int j = dcMonomialSize; j < nNN; j++)
                    {
                        // Id in the list
                        std::ptrdiff_t const Id = j * nDim;

                        // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                        poly.monomialValue(L1Dist.data() + Id, column);

                        EVectorMapType<T> columnV(column, dcMonomialSize);

                        // Fill the Vandermonde matrix column by column
                        VMTimage.block(0, j, dcMonomialSize, 1) << columnV;

                        // Neighbor point number
                        int const IdJ = NearestNeighbors[j];

                        /*
                         * Using a smooth correction function that satisfies
                         * \f$ {\mathbf F} \left(\frac{{\mathbf x}_p-{\mathbf x}_q}{c({\mathbf x}_q)} \right) =\delta_{pq} \f$
                         * Choose \f$ c({\mathbf x}) \f$ such that it is smaller than the distance
                         * between the point and its nearest neighbors
                         */
                        T s = std::sqrt(nnDist[j]) / (0.9 * std::sqrt(idataminDist[IdJ]));

                        // Compute the kernel value at the point
                        T dckernelV = q.f(&s);

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
                        EMimage(j) = std::exp(-nnDist[j] * byEpsilonsq2);
                    }

                    /* 
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMTimage = VMTimage * EMatrixX<T>(EMimage.asDiagonal());
                    AM = BMTimage * BMTimage.transpose();
                }
                else
                {
                    /* 
                     * \f$
                     * \begin{matrix} 
                     * {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix}
                     * \f$
                     */
                    BMT = VMT * EMatrixX<T>(EM.asDiagonal());
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
                for (int j = dcrank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t const Id = k * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<T> columnV(column, dcMonomialSize);

                    // Get the column l which causes singularity
                    columnL << VMT.block(0, l, dcMonomialSize, 1);

                    // Fill the Vandermonde matrix by the new column
                    VMT.block(0, l, dcMonomialSize, 1) << columnV;

                    // Neighbor point number
                    int const IdJ = NearestNeighbors[k];

                    /* 
                     * Using a smooth correction function that satisfies
                     * \f$ {\mathbf F} \left(\frac{{\mathbf x}_p-{\mathbf x}_q}{c({\mathbf x}_q)} \right) =\delta_{pq} \f$
                     * Choose \f$ c({\mathbf x}) \f$ such that it is smaller than the distance
                     * between the point and its nearest neighbors
                     */
                    T s = std::sqrt(nnDist[k]) / (0.9 * std::sqrt(idataminDist[IdJ]));

                    // Compute the kernel value at the point IdK
                    T dckernelV = q.f(&s);

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

                for (int j = dcrank, k = dcMonomialSize; j < dcMonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    EM(l) = std::exp(-nnDist[k] * byEpsilonsq2);
                }

                /* 
                 * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                 * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                 * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1} 
                 * \f}
                 */
                BMT = VMT * EMatrixX<T>(EM.asDiagonal());
                AM = BMT * BMT.transpose();
            }

            {
                /*
                 * Two-sided Jacobi SVD decomposition, ensuring optimal reliability and accuracy.
                 * Thin U and V are all we need for (least squares) solving.
                 */
                Eigen::JacobiSVD<EMatrixX<T>> svd(AM, Eigen::DecompositionOptions::ComputeThinU | Eigen::DecompositionOptions::ComputeThinV);

                /*
                 * SV contains the least-squares solution of 
                 * \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b} \f$
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

            if (dcrank < dcMonomialSize - nENN)
            {
                // Loop through the neighbors
                for (int j = 0; j < dcMonomialSize; j++)
                {
                    // Id in the list
                    std::ptrdiff_t const Id = j * nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<T> columnV(column, dcMonomialSize);

                    T const expo = std::exp(-nnDist[j] * byEpsilonsq);

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
                    T expo;

                    if (j >= dcrank)
                    {
                        // Id in the list
                        Id = m * nDim;
                        expo = std::exp(-nnDist[m] * byEpsilonsq);
                        m++;
                    }
                    else
                    {
                        Id = l * nDim;
                        expo = std::exp(-nnDist[l] * byEpsilonsq);
                    }

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomialValue(L1Dist.data() + Id, column);

                    EVectorMapType<T> columnV(column, dcMonomialSize);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + l;
                    dcKernel[IdK] += SV.dot(columnV) * expo;
                }

                // Loop through the neighbors
                for (int j = dcrank, m = dcMonomialSize; j < dcMonomialSize; j++, m++)
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
             * \f{matrix} {{\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
             * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
             * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1} 
             * \f}
             */
            BMT = VMT * EMatrixX<T>(EM.asDiagonal());

            AM = BMT * BMT.transpose();

            // SV contains the solution of \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b} \f$
            SV = AM.lu().solve(RHSB);

            // Loop through the neighbors
            for (int j = 0; j < dcMonomialSize; j++)
            {
                // Id in the list
                std::ptrdiff_t const Id = j * nDim;

                // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                poly.monomialValue(L1Dist.data() + Id, column);

                EVectorMapType<T> columnV(column, dcMonomialSize);

                T const expo = std::exp(-nnDist[j] * byEpsilonsq);

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

template <typename T, class Distance>
bool dcpse<T, Distance>::compute(T *iFvalue, int const nPoints, T *qFvalue, int const nqPoints)
{
    if (KNN->numInputdata() != nPoints)
    {
        UMUQFAILRETURN("Input data does not match with previously computed weights!");
    }
    if (KNN->numQuerydata() != nqPoints)
    {
        UMUQFAILRETURN("Query data does not match with previously computed weights!");
    }
    if (dcKernelSize != nqPoints * dcMonomialSize)
    {
        UMUQFAILRETURN("Previously computed weights does not match with this query data!");
    }

    if (qFvalue == nullptr)
    {
        try
        {
            qFvalue = new T[nqPoints];
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
}

template <typename T, class Distance>
bool dcpse<T, Distance>::interpolate(T *iFvalue, int const nPoints, T *&qFvalue, int const nqPoints)
{
    if (KNN->numInputdata() != nPoints)
    {
        UMUQFAILRETURN("Input data does not match with previously computed weights!");
    }
    if (KNN->numQuerydata() != nqPoints)
    {
        UMUQFAILRETURN("Query data does not match with previously computed weights!");
    }
    if (dcKernelSize != nqPoints * dcMonomialSize)
    {
        UMUQFAILRETURN("Previously computed weights does not match with this query data!");
    }

    if (qFvalue == nullptr)
    {
        try
        {
            qFvalue = new T[nqPoints];
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    // Loop over all query points
    for (int i = 0; i < nqPoints; i++)
    {
        // A pointer to nearest neighbors indices of point i
        int *NearestNeighbors = KNN->NearestNeighbors(i);

        int IdI = i * dcMonomialSize;

        T sum(0);

        // std::cout << "For point i=" << i << " dcKernel=";
        // Loop through the neighbors
        for (int j = 0; j < dcMonomialSize; j++, IdI++)
        {
            int const IdJ = NearestNeighbors[j];
            sum += dcKernel[IdI] * iFvalue[IdJ];

            // std::cout << dcKernel[IdI] << " ";
        }
        // std::cout << "Fvalue=" << sum << " h_average=" << h_average[i] << std::endl;
        qFvalue[i] = sum;
    }

    return true;
}

template <typename T, class Distance>
inline T *dcpse<T, Distance>::neighborhoodKernel(int const index) const
{
    return dcKernel.data() + index * dcMonomialSize;
}

template <typename T, class Distance>
inline T *dcpse<T, Distance>::neighborhoodKernel() const
{
    return dcKernel.data();
}

template <typename T, class Distance>
inline int dcpse<T, Distance>::neighborhoodKernelSize() const
{
    return dcMonomialSize;
}

template <typename T, class Distance>
inline int dcpse<T, Distance>::orderofAccuracy(int const index) const
{
    return Order[index];
}

template <typename T, class Distance>
inline void dcpse<T, Distance>::printInfo() const
{
    for (int i = 0; i < nTerms; i++)
    {
        std::cout << Order[i] << (Order[i] % 10 == 1 ? "st " : Order[i] % 10 == 2 ? "nd " : Order[i] % 10 == 3 ? "rd " : "th ") << "order DC-PSE kernel uses \n"
                  << neighborhoodKernelSize() << " points in the neighborhood of each query points." << std::endl;
    }
}

template <typename T, class Distance>
inline T dcpse<T, Distance>::averageSpace(int const index) const
{
    return h_average[index];
}

template <typename T, class Distance>
inline T *dcpse<T, Distance>::averageSpace() const
{
    return h_average.data();
}

} // namespace umuq

#endif // UMUQ_DCPSE

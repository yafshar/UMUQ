#ifndef UMUQ_DCPSE_H
#define UMUQ_DCPSE_H

#include "polynomial.hpp"
#include "factorial.hpp"
#include "eigenmatrix.hpp"
#include "knearestneighbors.hpp"
#include "primitive.hpp"

/*! \class dcpse
 * \brief This is a general class for (DC-PSE)
 * 
 * It creates a discretized differential operator and interpolators
 * 
 * \tparam T Data type
 */

// TODO : Currently the class works only for one term and it should be extended to multi terms

template <typename T>
class dcpse
{
  public:
    /*!
     * \brief Default constructor
     * 
     * \param ndim             Dimensiononality
     * \param nterms           Number of terms (currently only one term is implemented)
     */
    explicit dcpse(int ndim, int nterms = 1);

    /*!
     * \brief Move constructor, construct a new dcpse object from input dcpse object
     * 
     * \param inputDC 
     */
    explicit dcpse(dcpse<T> &&inputDC);

    /*!
     * \brief Move assignment operator
     * 
     * \param inputDB 
     * \return dcpse<T>& 
     */
    dcpse<T> &operator=(dcpse<T> &&inputDC);

    /*!
     * \brief Destroy the dcpse object
     * 
     */
    ~dcpse() {}

    /*! \fn ComputeWeights
     * \brief Computes generalized DC-PSE differential operators on set of input points
     *
     * This function uses one set of points as input data to compute the generalized DC-PSE 
     * differential operators.
     * If the degree of the differential operator is zero \f$ | \beta | = 0 \f$, suggests one 
     * should use the interpolator function not this one. 
     * 
     * \param idata    A pointer to input data
     * \param nPoints  Number of data points
     * \param beta     In multi-dimensional notation \f$ \beta=\left(\beta_1, \cdots, \beta_d \right) \f$
     *                 Notation for partial derivatives:
     *                 \f[ 
     *                   \begin{align} D^\beta = \frac{\partial^{|\beta|}} {\partial x_1^{\beta_1} 
     *                   \partial x_2^{\beta_2}\cdots\partial x_d^{\beta_d}}. \label{eq:1} \end{align} 
     *                 \f]
     * \param order    Order of accuracy (default is 2nd order accurate)
     * \param nENN     Number of extra nearest neighbors to aid in case of sigularity of the Vandermonde matrix (default is 2)
     * \param ratio    The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     * 
     */
    bool computeWeights(T *idata, int const nPoints, int *beta, int order = 2, int nENN = 2, T ratio = static_cast<T>(1));

    /*! \fn computeWeights
     * \brief Computes generalized DC-PSE differential operators on the set of query points.
     * 
     * This function uses one set of points as input data to compute the generalized DC-PSE 
     * differential opearators on the set of query points.
     * If the degree of the differential operator is zero \f$ | \beta | = 0 \f$, means one should
     * use the interpolator function not this one. 
     * 
     * \param idata     A pointer to input data
     * \param nPoints   Number of data points
     * \param qdata     A pointer to query data
     * \param nqPoints  Number of query data points
     * \param beta      In multi-dimensional notation \f$ \beta=\left(\beta_1, \cdots, \beta_d \right) \f$
     *                  Notation for partial derivatives:
     *                  \f$ \begin{align} D^\beta = \frac{\partial^{|\beta|}} {\partial x_1^{\beta_1} \partial x_2^{\beta_2}\cdots\partial x_d^{\beta_d}}. \label{eq:1} \end{align} \f$
     * \param order     Order of accuracy (default is 2nd order accurate)
     * \param nENN      Number of extra nearest neighbors to aid in case of sigularity of the Vandermonde matrix (default is 2)
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
     * \param nENN     Number of extra nearest neighbors to aid in case of sigularity of the Vandermonde matrix (default is 2)
     * \param ratio    The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     */
    bool computeInterpolatorWeights(T *idata, int const nPoints, int order = 2, int nENN = 2, T ratio = static_cast<T>(1));

    /*! \fn computeInterpolatorWeights
     * \brief Computes generalized DC-PSE interpolator operators on the set of query points.
     * 
     * This function uses one set of points as input data to compute the generalized DC-PSE 
     * interpolator operators on the set of query points.
     * 
     * \param idata            A pointer to input data 
     * \param nPoints          Number of data points
     * \param qdata            A pointer to query data 
     * \param nqPoints         Number of query data points
     * \param order            Order of accuracy (default is 2nd order accurate)
     * \param nENN             Number of extra nearest neighbors to aid in case of sigularity of the Vandermonde matrix (default is 2)
     * \param ratio            The \f$ \frac{h}{\epsilon} \f$ the default vale is one
     */
    bool computeInterpolatorWeights(T *idata, int const nPoints, T *qdata, int const nqPoints, int order = 2, int nENN = 2, T ratio = static_cast<T>(1));

    /*! \fn compute
     * \brief Evaluate a discretized DC-PSE operator from function values of input data and put the results as the query data function values
     * 
     * This function uses function values of input data and the weights of the operator which have 
     * been previously computed to compute the query values and put the results as the query data 
     * function values. 
     * 
     * At first it checks the computed kernel size to be equal to the number of query points times the 
     * size of monomials which has been previously computed for the required degree of the DC-PSE operator.
     * 
     * \param iFvalue   A pointer to input data function value
     * \param nPoints   Number of data points
     * \param qFvalue   A pointer to query data function value
     * \param nqPoints  Number of query data points
     */
    bool compute(T *iFvalue, int const nPoints, T *qFvalue, int const nqPoints);

    /*! \fn interpolate
     * \brief Evaluate a discretized DC-PSE interpolation operator from function values of input data and put the 
     * interpolation results as the query data values
     * 
     * This function uses function values of input data and the weights of the interpolation operator which have 
     * been previously computed to compute the query values and put the results as the query data 
     * function values. 
     * 
     * At first it checks the computed kernel size to be equal to the number of query points times the 
     * size of monomials which has been previously computed for the required degree of DC-PSE operator
     * or interpolator.
     * 
     * \param iFvalue   A pointer to input data function value
     * \param nPoints   Number of data points
     * \param qFvalue   A pointer to query data function value
     * \param nqPoints  Number of query data points
     */
    bool interpolate(T *iFvalue, int const nPoints, T *qFvalue, int const nqPoints);

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
     * \returns  A pointer to kernel array of all query points
     */
    inline T *neighborhoodKernel() const;

    /*!
     * \brief   Size of the neighborhood kernel which equals to the monomial size 
     * 
     * \returns Size of the neighborhood kernel
     */
    inline int neighborhoodKernelSize() const;

    /*!
     * \brief order of accuracy of DC-PSE kernel at index
     * 
     * \param  index Index number in nTerms array
     * \return order of accuracy of DC-PSE kernel at index
     */
    inline int orderofAccuracy(int const index = 0) const;

    /*!
     * \brief print the DC-PSE information
     * 
     */
    inline void printInfo() const;

    /*!
     * \brief Component-wise average neighbor spacing @index
     * 
     * \param index Index of a point (from data points) to get its average neighbor spacing
     * 
     * \returns A component-wise average neighbor spacing @index 
     */
    inline T averageSpace(int const index) const;

    /*!
     * \brief A pointer to component-wise average neighbor spacing
     * 
     * \returns A pointer to component-wise average neighbor spacing
     */
    inline T *averageSpace() const;

  private:
    // Make it noncopyable
    dcpse(dcpse<T> const &) = delete;

    // Make it not assignable
    dcpse<T> &operator=(dcpse<T> const &) = delete;

  private:
    //! Dimensionality
    int nDim;

    //! Number of terms
    int nTerms;

    //! The monomial size
    /* 
     * \f$ monomialSize = \left(\begin{matrix} r + d -1 \\ d \end{matrix}\right) \f$
     */
    int dcmonomialSize;

    //! Size of the kernel
    int dckernelSize;

    //! Order of accuracy for each term
    std::unique_ptr<int[]> Order;

    //! Operator kernel
    std::unique_ptr<T[]> dckernel;

    //! k-NearestNeighbor Object
    std::unique_ptr<L2NearestNeighbor<T>> KNN;

    //! Component-wise average neighbor spacing \f$ h = \frac{1}{N} \sum_{p=1}^{N}\left(|x_{1}-x_{p1}| + \cdots |x_{d} -x_{pd}| \right), \f$
    std::vector<T> h_average;

    //! The sign is chosen positive for odd \f$ | \beta | \f$ and negative for even \f$ | \beta | \f$
    T rhscoeff;
};

/*!
 * \brief Default constructor
 * 
 * \param ndim             Dimensiononality
 * \param nterms           Number of terms (currently only one term is implemented)
 */
template <typename T>
dcpse<T>::dcpse(int ndim, int nterms) : nDim(ndim),
                                        nTerms(nterms),
                                        dcmonomialSize(0),
                                        dckernelSize(0),
                                        Order{new int[nterms]} {}

/*!
 * \brief Move constructor, construct a new dcpse object from input dcpse object
 * 
 * \param inputDC 
 */
template <typename T>
dcpse<T>::dcpse(dcpse<T> &&inputDC)
{
    this->nDim = inputDC.nDim;
    this->nTerms = inputDC.nTerms;
    this->dcmonomialSize = inputDC.dcmonomialSize;
    this->dckernelSize = inputDC.dckernelSize;
    this->Order = std::move(inputDC.Order);
    this->dckernel = std::move(inputDC.dckernel);
    this->KNN = std::move(inputDC.KNN);
    this->h_average = std::move(inputDC.h_average);
    this->rhscoeff = inputDC.rhscoeff;
}

/*!
 * \brief Move assignment operator
 * 
 * \param inputDB 
 * \return dcpse<T>& 
 */
template <typename T>
dcpse<T> &dcpse<T>::operator=(dcpse<T> &&inputDC)
{
    this->nDim = inputDC.nDim;
    this->nTerms = inputDC.nTerms;
    this->dcmonomialSize = inputDC.dcmonomialSize;
    this->dckernelSize = inputDC.dckernelSize;
    this->Order = std::move(inputDC.Order);
    this->dckernel = std::move(inputDC.dckernel);
    this->KNN = std::move(inputDC.KNN);
    this->h_average = std::move(inputDC.h_average);
    this->rhscoeff = inputDC.rhscoeff;

    return *this;
}

/*! \fn ComputeWeights
 * \brief Computes generalized DC-PSE differential operators on set of input points
 *
 * This function uses one set of points as input data to compute the generalized DC-PSE 
 * differential operators.
 * If the degree of the differential operator is zero \f$ | \beta | = 0 \f$, suggests one 
 * should use the interpolator function not this one. 
 * 
 * \param idata    A pointer to input data
 * \param nPoints  Number of data points
 * \param beta     In multi-dimensional notation \f$ \beta=\left(\beta_1, \cdots, \beta_d \right) \f$
 *                 Notation for partial derivatives:
 *                 \f[ 
 *                      \begin{align} D^\beta = \frac{\partial^{|\beta|}} {\partial x_1^{\beta_1} 
 *                      \partial x_2^{\beta_2}\cdots\partial x_d^{\beta_d}}. \label{eq:1} \end{align} 
 *                  \f]
 * \param order    Order of accuracy (default is 2nd order accurate)
 * \param nENN     Number of extra nearest neighbors to aid in case of sigularity of the Vandermonde matrix (default is 2)
 * \param ratio    The \f$ \frac{h}{\epsilon} \f$ the default vale is one
 * 
 */
template <typename T>
bool dcpse<T>::computeWeights(T *idata, int const nPoints, int *beta, int order, int nENN, T ratio)
{
    if (nPoints < 1)
    {
        UMUQFAILRETURN("Number of input data points are negative!");
    }

    // Extra check on the order
    order = (order > 0) ? order : 2;
    {
        int *o = this->Order.get();
        std::fill(o, o + this->nTerms, order);
    }

    // Extra check on the number of extra nearest neighbors
    nENN = (nENN > 0) ? nENN : 0;

    // Extra check on the ratio
    ratio = (ratio > 0) ? ratio : static_cast<T>(1);

    // \f$ |\beta| = \beta_1 + \cdots + \beta_d \f$
    int Beta = std::accumulate(beta, beta + this->nDim, 0);
    if (Beta == 0)
    {
        UMUQWARNING("Zero order degree derivative gives an approximation! \n If this is an interpolation use the interpolation function!");
    }

    int alphamin = (Beta % 2 == 0);

    // \f$ (-1)^{|\beta|} \f$
    this->rhscoeff = alphamin ? static_cast<T>(1) : -static_cast<T>(1);

    // Create an instance of polynomial object with polynomial degree of \f$ |\beta| + r -1 \f$
    polynomial<T> poly(this->nDim, order + Beta - 1);

    /*
     * Get the monomials size
     * \f$ monomialSize = \left(\begin{matrix} |\beta| + r + d -1 \\ d \end{matrix}\right) - \alpha_{\min} \f$
     */
    this->dcmonomialSize = poly.monomialsize() - alphamin;

    if (nPoints * this->dcmonomialSize > this->dckernelSize)
    {
        this->dckernelSize = nPoints * this->dcmonomialSize;
        try
        {
            // Make sure of the correct kernel size
            this->dckernel.reset(new T[this->dckernelSize]);

            this->h_average.resize(nPoints);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        this->dckernelSize = nPoints * this->dcmonomialSize;
    }

    if (this->KNN)
    {
        if (nPoints != this->KNN->numInputdata() || nPoints != this->KNN->numQuerydata())
        {
            try
            {
                /*
                 * Finding K nearest neighbors
                 * The number of points K in the neighborhood of each point
                 * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
                 */
                this->KNN.reset(new L2NearestNeighbor<T>(nPoints, this->nDim, this->dcmonomialSize + nENN));
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
            this->KNN.reset(new L2NearestNeighbor<T>(nPoints, this->nDim, this->dcmonomialSize + nENN));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    // Construct a kd-tree index & do nearest neighbors search
    this->KNN->buildIndex(idata);

    /*
     * Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
     * \f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b}  \f$
     */
    EVectorX<T> RHSB(this->dcmonomialSize);
    {
        // Get a pointer to the monomial basis
        int *alpha = poly.monomial_basis();

        for (int i = 0, id = alphamin * this->nDim; i < this->dcmonomialSize; i++)
        {
            int maxalpha = 0;
            for (int j = 0; j < this->nDim; j++, id++)
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
                std::for_each(beta, beta + this->nDim, [&](int const b_j) { fact *= factorial<T>(b_j); });
                RHSB(i) = this->rhscoeff * fact;
            }
        }

        // TODO : check this again
        /* 
         * When applicable, and for stability reasons, set the zeroth moment to 5
         */
        if (this->rhscoeff > T{})
        {
            std::ptrdiff_t const id = alphamin * this->nDim;
            if (std::accumulate(alpha + id, alpha + id + this->nDim, 0) == 0)
            {
                RHSB(0) = static_cast<T>(5);
            }
        }
    }

    // Total number of nearest neighbours for each point
    int nNN = this->KNN->numNearestNeighbors();

    /*
     * Creating a transpose of the Vandermonde matrix
     * with the size of monomials * monomials \f$  = l \times l \f$
     */
    EMatrixX<T> VMT(this->dcmonomialSize, this->dcmonomialSize);
    EMatrixX<T> VMTimage(this->dcmonomialSize, nNN);

    // Matrix of exponential window function
    EVectorX<T> EM(this->dcmonomialSize);
    EVectorX<T> EMimage(nNN);

    // Matrix A of a linear system for the kernel coefficients
    EMatrixX<T> AM(this->dcmonomialSize, this->dcmonomialSize);

    // Matrix B
    EMatrixX<T> BMT(this->dcmonomialSize, this->dcmonomialSize);
    EMatrixX<T> BMTimage(this->dcmonomialSize, nNN);

    // ${\mathbf a}^T({\mathbf x})$ is the column vector of coefficients which is the solution of linear system
    EVectorX<T> SV(this->dcmonomialSize);

    // Array for keeping the component-wise L1 distances
    std::vector<T> L1Dist(nNN * this->nDim);

    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
    T *column = new T[this->dcmonomialSize + alphamin];

    std::vector<int> IndexId(this->dcmonomialSize);

    // Loop over all points
    for (int i = 0; i < nPoints; i++)
    {
        std::ptrdiff_t const IdM = i * this->dcmonomialSize;
        std::ptrdiff_t const IdI = i * this->nDim;

        // A pointer to nearest neighbors indices of point i
        int *NearestNeighbors = this->KNN->NearestNeighbors(i);

        // A pointer to nearest neighbors distances from the point i
        T *nnDist = this->KNN->NearestNeighborsDistances(i);

        /* 
         * For each point \f$ {\mathbf x} \f$ we define:
         * 
         * \[f
         * \left\{{\mathbf z}_p({\mathbf x}) \right\}_{p=1}^{k} = \left\{{\mathbf x}_p - {\mathbf x} \right\}, 
         * \f]
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
                std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->nDim;

                // pointer to idata (neighbors of i)
                T *Jdata = idata + IdJ;

                for (int d = 0; d < this->nDim; d++, n++)
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

        this->h_average[i] = h_avg;

        // Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
        T const byEpsilon = ratio / h_avg;
        T const byEpsilonsq = byEpsilon * byEpsilon;
        T const byEpsilonsq2 = 0.5 * byEpsilonsq;
        T const byEpsilonPowerBeta = std::pow(byEpsilon, Beta);

        // Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](T &l_i) { l_i *= byEpsilon; });

        // Loop through the neighbors
        for (int j = 0; j < this->dcmonomialSize; j++)
        {
            // Id in the L1 distance list
            std::ptrdiff_t const Id = j * this->nDim;

            // Evaluates a monomial at a point \f$ {\mathbf x} \f$
            poly.monomial_value(L1Dist.data() + Id, column);

            TEMapVectorX<T> columnV(column + alphamin, this->dcmonomialSize);

            // Fill the Vandermonde matrix column by column
            VMT.block(0, j, this->dcmonomialSize, 1) << columnV;
        }

        for (int j = 0; j < this->dcmonomialSize; j++)
        {
            EM(j) = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq2);
        }

        int dcrank;

        {
            // LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<T>> lu(VMT);

            dcrank = lu.rank();

            if (dcrank < this->dcmonomialSize && dcrank >= this->dcmonomialSize - nENN)
            {
                for (int j = 0; j < this->dcmonomialSize; j++)
                {
                    IndexId[j] = lu.permutationQ().indices()(j);
                }
            }
        }

        if (dcrank < this->dcmonomialSize)
        {
            UMUQWARNING("There are some singularities! we use a least-squares solution!");

            // If necessary, remove redundant equations/coefficients

            // Number of neighbor points are not enough
            if (dcrank < this->dcmonomialSize - nENN)
            {
                UMUQWARNING("Number of neighbor points are not enough! Matrix rank =");
                std::cerr << dcrank << " < " << this->dcmonomialSize - nENN << std::endl;

                if (nENN > 0)
                {
                    VMTimage.block(0, 0, this->dcmonomialSize, this->dcmonomialSize) << VMT;
                    EMimage.head(this->dcmonomialSize) << EM;

                    // Loop through the rest of nearest neighbors
                    for (int j = this->dcmonomialSize; j < nNN; j++)
                    {
                        // Id in the list
                        std::ptrdiff_t const Id = j * this->nDim;

                        // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                        poly.monomial_value(L1Dist.data() + Id, column);

                        TEMapVectorX<T> columnV(column + alphamin, this->dcmonomialSize);

                        // Fill the Vandermonde matrix column by column
                        VMTimage.block(0, j, this->dcmonomialSize, 1) << columnV;
                    }

                    for (int j = this->dcmonomialSize; j < nNN; j++)
                    {
                        EMimage(j) = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq2);
                    }

                    /* 
                     * \f[ 
                     * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix} 
                     * \f]
                     */
                    BMTimage = VMTimage * EMatrixX<T>(EMimage.asDiagonal());
                    AM = BMTimage * BMTimage.transpose();
                }
                else
                {
                    /* 
                     * \f[ 
                     * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix} 
                     * \f]
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
                for (int j = dcrank, k = this->dcmonomialSize; j < this->dcmonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t const Id = k * this->nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomial_value(L1Dist.data() + Id, column);

                    TEMapVectorX<T> columnV(column + alphamin, this->dcmonomialSize);

                    // Fill the Vandermonde matrix by the new column
                    VMT.block(0, l, this->dcmonomialSize, 1) << columnV;
                }

                for (int j = dcrank, k = this->dcmonomialSize; j < this->dcmonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    EM(l) = std::exp(-nnDist[k] * nnDist[k] * byEpsilonsq2);
                }

                /*
                 * \f[
                 *  \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                 *  {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                 *  {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                 *  \end{matrix}
                 * \f]
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

            // TODO: Correct IndexId in the case of SVD. Right now, this is the best I can do
            /*
             * Later I should check on SVD solution and to find out which columns are the
             * Most important one, then I can correct the IndexId order
             */
            if (dcrank < this->dcmonomialSize - nENN)
            {
                // Loop through the neighbors
                for (int j = 0; j < this->dcmonomialSize; j++)
                {
                    // Id in the list
                    std::ptrdiff_t const Id = j * this->nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomial_value(L1Dist.data() + Id, column);

                    TEMapVectorX<T> columnV(column + alphamin, this->dcmonomialSize);

                    T const expo = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + j;
                    this->dckernel[IdK] = SV.dot(columnV) * byEpsilonPowerBeta * expo;
                }
            }
            else
            {
                // Loop through the neighbors
                for (int j = 0, m = this->dcmonomialSize; j < this->dcmonomialSize; j++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t Id;
                    T expo;

                    if (j >= dcrank)
                    {
                        // Id in the list
                        Id = m * this->nDim;
                        expo = std::exp(-nnDist[m] * nnDist[m] * byEpsilonsq);
                        m++;
                    }
                    else
                    {
                        Id = l * this->nDim;
                        expo = std::exp(-nnDist[l] * nnDist[l] * byEpsilonsq);
                    }

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomial_value(L1Dist.data() + Id, column);

                    TEMapVectorX<T> columnV(column + alphamin, this->dcmonomialSize);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + l;
                    this->dckernel[IdK] = SV.dot(columnV) * byEpsilonPowerBeta * expo;
                }

                // Loop through the neighbors
                for (int j = dcrank, m = this->dcmonomialSize; j < this->dcmonomialSize; j++, m++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Correct the neighborhood order
                    this->KNN->IndexSwap(l, m);
                }
            }
        }
        else
        {
            /* 
             * \f[ 
             * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
             * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
             * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
             * \end{matrix} 
             * \f]
             */
            BMT = VMT * EMatrixX<T>(EM.asDiagonal());

            AM = BMT * BMT.transpose();

            // SV contains the solution of \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b} \f$
            SV = AM.lu().solve(RHSB);

            // Loop through the neighbors
            for (int j = 0; j < this->dcmonomialSize; j++)
            {
                // Id in the list
                std::ptrdiff_t const Id = j * this->nDim;

                // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                poly.monomial_value(L1Dist.data() + Id, column);

                TEMapVectorX<T> columnV(column + alphamin, this->dcmonomialSize);

                T const expo = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq);

                // Index inside the kernel
                std::ptrdiff_t const IdK = IdM + j;
                this->dckernel[IdK] = SV.dot(columnV) * byEpsilonPowerBeta * expo;
            }
        }
    } // Loop over all points

    delete[] column;

    return true;
}

/*! \fn computeWeights
 * \brief Computes generalized DC-PSE differential operators on the set of query points.
 * 
 * This function uses one set of points as input data to compute the generalized DC-PSE 
 * differential opearators on the set of query points.
 * If the degree of the differential operator is zero \f$ | \beta | = 0 \f$, means one should
 * use the interpolator function not this one. 
 * 
 * \param idata     A pointer to input data
 * \param nPoints   Number of data points
 * \param qdata     A pointer to query data
 * \param nqPoints  Number of query data points
 * \param beta      In multi-dimensional notation \f$ \beta=\left(\beta_1, \cdots, \beta_d \right) \f$
 *                  Notation for partial derivatives:
 *                  \f$ \begin{align} D^\beta = \frac{\partial^{|\beta|}} {\partial x_1^{\beta_1} \partial x_2^{\beta_2}\cdots\partial x_d^{\beta_d}}. \label{eq:1} \end{align} \f$
 * \param order     Order of accuracy (default is 2nd order accurate)
 * \param nENN      Number of extra nearest neighbors to aid in case of sigularity of the Vandermonde matrix (default is 2)
 * \param ratio     The \f$ \frac{h}{\epsilon} \f$ the default vale is one
 */
template <typename T>
bool dcpse<T>::computeWeights(T *idata, int const nPoints, T *qdata, int const nqPoints, int *beta, int order, int nENN, T ratio)
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
        int *o = this->Order.get();
        std::fill(o, o + this->nTerms, order);
    }

    // Extra check on the number of extra nearest neighbors
    nENN = (nENN > 0) ? nENN : 0;

    // Extra check on the ratio
    ratio = (ratio > 0) ? ratio : static_cast<T>(1);

    // \f$ |\beta| = \beta_1 + \cdots + \beta_d \f$
    int Beta = std::accumulate(beta, beta + this->nDim, 0);
    if (Beta == 0)
    {
        UMUQWARNING("Zero order degree derivative gives an approximation! \n If this is an interpolation use the interpolation function!");
    }

    int alphamin = (Beta % 2 == 0);

    // \f$ (-1)^{|\beta|} \f$
    this->rhscoeff = alphamin ? static_cast<T>(1) : -static_cast<T>(1);

    // Create an instance of polynomial object with polynomial degree of \f$ |\beta| + r -1 \f$
    polynomial<T> poly(this->nDim, order + Beta - 1);

    /*
     * Get the monomials size
     * \f$ monomialSize = \left(\begin{matrix} |\beta| + r + d -1 \\ d \end{matrix}\right) - \alpha_{\min} \f$
     */
    this->dcmonomialSize = poly.monomialsize() - alphamin;

    if (nqPoints * this->dcmonomialSize > this->dckernelSize)
    {
        this->dckernelSize = nqPoints * this->dcmonomialSize;
        try
        {
            // Make sure of the correct kernel size
            this->dckernel.reset(new T[this->dckernelSize]);

            this->h_average.resize(nqPoints);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        this->dckernelSize = nqPoints * this->dcmonomialSize;
    }

    if (this->KNN)
    {
        if (nqPoints != this->KNN->numInputdata() || nqPoints != this->KNN->numQuerydata())
        {
            try
            {
                /*
                 * Finding K nearest neighbors
                 * The number of points K in the neighborhood of each point
                 * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
                 */
                this->KNN.reset(new L2NearestNeighbor<T>(nPoints, nqPoints, this->nDim, this->dcmonomialSize + nENN));
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
            this->KNN.reset(new L2NearestNeighbor<T>(nPoints, nqPoints, this->nDim, this->dcmonomialSize + nENN));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    // Construct a kd-tree index & do nearest neighbors search
    this->KNN->buildIndex(idata, qdata);

    /*
     * Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
     * \f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b}  \f$
     */
    EVectorX<T> RHSB(this->dcmonomialSize);
    {
        // Get a pointer to the monomial basis
        int *alpha = poly.monomial_basis();

        for (int i = 0, id = alphamin * this->nDim; i < this->dcmonomialSize; i++)
        {
            int maxalpha = 0;
            for (int j = 0; j < this->nDim; j++, id++)
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
                std::for_each(beta, beta + this->nDim, [&](int const b_j) { fact *= factorial<T>(b_j); });
                RHSB(i) = this->rhscoeff * fact;
            }
        }

        // TODO : check this again
        /* 
         * At off-particle locations it should be always zero to obtain kernels
         * with a vanishing zeroth-order moment that can be consistently evaluated
         */
        RHSB(0) = T{};
    }

    // Total number of nearest neighbours for each point
    int nNN = this->KNN->numNearestNeighbors();

    // Array for keeping the component-wise L1 distances
    std::vector<T> L1Dist(nNN * this->nDim);

    /*
     * Creating a transpose of the Vandermonde matrix
     * with the size of monomials * monomials \f$  = l \times l \f$
     */
    EMatrixX<T> VMT(this->dcmonomialSize, this->dcmonomialSize);
    EMatrixX<T> VMTimage(this->dcmonomialSize, nNN);

    // Matrix of exponential window function
    EVectorX<T> EM(this->dcmonomialSize);
    EVectorX<T> EMimage(nNN);

    // Matrix A of a linear system for the kernel coefficients
    EMatrixX<T> AM(this->dcmonomialSize, this->dcmonomialSize);

    // Matrix B
    EMatrixX<T> BMT(this->dcmonomialSize, this->dcmonomialSize);
    EMatrixX<T> BMTimage(this->dcmonomialSize, nNN);

    // ${\mathbf a}^T({\mathbf x})$ is the column vector of coefficients which is the solution of linear system
    EVectorX<T> SV(this->dcmonomialSize);

    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
    T *column = new T[this->dcmonomialSize + alphamin];

    std::vector<int> IndexId(this->dcmonomialSize);

    // Loop over all query points
    for (int i = 0; i < nqPoints; i++)
    {
        std::ptrdiff_t const IdM = i * this->dcmonomialSize;
        std::ptrdiff_t const IdI = i * this->nDim;

        // A pointer to nearest neighbors indices of point i
        int *NearestNeighbors = this->KNN->NearestNeighbors(i);

        // A pointer to nearest neighbors distances from the point i
        T *nnDist = this->KNN->NearestNeighborsDistances(i);

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
                std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->nDim;

                // pointer to idata (neighbors of i)
                T *Jdata = idata + IdJ;

                for (int d = 0; d < this->nDim; d++, n++)
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

        this->h_average[i] = h_avg;

        // Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
        T const byEpsilon = ratio / h_avg;
        T const byEpsilonsq = byEpsilon * byEpsilon;
        T const byEpsilonsq2 = 0.5 * byEpsilonsq;
        T const byEpsilonPowerBeta = std::pow(byEpsilon, Beta);

        // Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](T &l_i) { l_i *= byEpsilon; });

        // Loop through the neighbors
        for (int j = 0; j < this->dcmonomialSize; j++)
        {
            // Id in the L1 distance list
            std::ptrdiff_t const Id = j * this->nDim;

            // Evaluates a monomial at a point \f$ {\mathbf x} \f$
            poly.monomial_value(L1Dist.data() + Id, column);

            TEMapVectorX<T> columnV(column + alphamin, this->dcmonomialSize);

            // Fill the Vandermonde matrix column by column
            VMT.block(0, j, this->dcmonomialSize, 1) << columnV;
        }

        for (int j = 0; j < this->dcmonomialSize; j++)
        {
            EM(j) = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq2);
        }

        int dcrank;

        {
            // LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<T>> lu(VMT);

            dcrank = lu.rank();

            if (dcrank < this->dcmonomialSize && dcrank >= this->dcmonomialSize - nENN)
            {
                for (int j = 0; j < this->dcmonomialSize; j++)
                {
                    IndexId[j] = lu.permutationQ().indices()(j);
                }
            }
        }

        if (dcrank < this->dcmonomialSize)
        {
            UMUQWARNING("There are some singularities! we use a least-squares solution!");

            // if necessary, remove redundant equations/coefficients

            // Number of neighbor points are not enough
            if (dcrank < this->dcmonomialSize - nENN)
            {
                UMUQWARNING("Number of neighbor points are not enough! Matrix rank = ");
                std::cerr << dcrank << " < " << this->dcmonomialSize - nENN << std::endl;

                if (nENN > 0)
                {
                    VMTimage.block(0, 0, this->dcmonomialSize, this->dcmonomialSize) << VMT;
                    EMimage.head(this->dcmonomialSize) << EM;

                    // Loop through the rest of nearest neighbors
                    for (int j = this->dcmonomialSize; j < nNN; j++)
                    {
                        // Id in the list
                        std::ptrdiff_t const Id = j * this->nDim;

                        // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                        poly.monomial_value(L1Dist.data() + Id, column);

                        TEMapVectorX<T> columnV(column + alphamin, this->dcmonomialSize);

                        // Fill the Vandermonde matrix column by column
                        VMTimage.block(0, j, this->dcmonomialSize, 1) << columnV;
                    }

                    for (int j = this->dcmonomialSize; j < nNN; j++)
                    {
                        EMimage(j) = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq2);
                    }

                    /* 
                     * \f[ 
                     * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix} 
                     * \f]
                     */
                    BMTimage = VMTimage * EMatrixX<T>(EMimage.asDiagonal());
                    AM = BMTimage * BMTimage.transpose();
                }
                else
                {
                    /* 
                     * \f[ 
                     * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix} 
                     * \f]
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
                for (int j = dcrank, k = this->dcmonomialSize; j < this->dcmonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t const Id = k * this->nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomial_value(L1Dist.data() + Id, column);

                    TEMapVectorX<T> columnV(column + alphamin, this->dcmonomialSize);

                    // Fill the Vandermonde matrix by the new column
                    VMT.block(0, l, this->dcmonomialSize, 1) << columnV;
                }

                for (int j = dcrank, k = this->dcmonomialSize; j < this->dcmonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    EM(l) = std::exp(-nnDist[k] * nnDist[k] * byEpsilonsq2);
                }

                /* 
                 * \f[ 
                 * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                 * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                 * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                 * \end{matrix} 
                 * \f]
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

            // TODO: Correct IndexId in the case of SVD. Right now, this is the best I can do
            /*
             * Later I should check on SVD solution and to find out which columns are the
             * Most important one, then I can correct the IndexId order
             */

            if (dcrank < this->dcmonomialSize - nENN)
            {
                // Loop through the neighbors
                for (int j = 0; j < this->dcmonomialSize; j++)
                {
                    // Id in the list
                    std::ptrdiff_t const Id = j * this->nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomial_value(L1Dist.data() + Id, column);

                    TEMapVectorX<T> columnV(column + alphamin, this->dcmonomialSize);

                    T const expo = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + j;
                    this->dckernel[IdK] = SV.dot(columnV) * byEpsilonPowerBeta * expo;
                }
            }
            else
            {
                // Loop through the neighbors
                for (int j = 0, m = this->dcmonomialSize; j < this->dcmonomialSize; j++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t Id;
                    T expo;

                    if (j >= dcrank)
                    {
                        // Id in the list
                        Id = m * this->nDim;
                        expo = std::exp(-nnDist[m] * nnDist[m] * byEpsilonsq);
                        m++;
                    }
                    else
                    {
                        Id = l * this->nDim;
                        expo = std::exp(-nnDist[l] * nnDist[l] * byEpsilonsq);
                    }

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomial_value(L1Dist.data() + Id, column);

                    TEMapVectorX<T> columnV(column + alphamin, this->dcmonomialSize);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + l;
                    this->dckernel[IdK] = SV.dot(columnV) * byEpsilonPowerBeta * expo;
                }

                // Loop through the neighbors
                for (int j = dcrank, m = this->dcmonomialSize; j < this->dcmonomialSize; j++, m++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Correct the neighborhood order
                    this->KNN->IndexSwap(l, m);
                }
            }
        }
        else
        {
            /* 
             * \f[ 
             * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
             * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
             * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
             * \end{matrix} 
             * \f]
             */
            BMT = VMT * EMatrixX<T>(EM.asDiagonal());

            AM = BMT * BMT.transpose();

            // SV contains the solution of \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b} \f$
            SV = AM.lu().solve(RHSB);

            // Loop through the neighbors
            for (int j = 0; j < this->dcmonomialSize; j++)
            {
                // Id in the list
                std::ptrdiff_t const Id = j * this->nDim;

                // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                poly.monomial_value(L1Dist.data() + Id, column);

                TEMapVectorX<T> columnV(column + alphamin, this->dcmonomialSize);

                T const expo = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq);

                // Index inside the kernel
                std::ptrdiff_t const IdK = IdM + j;
                this->dckernel[IdK] = SV.dot(columnV) * byEpsilonPowerBeta * expo;
            }
        }
    } // Loop over all points

    delete[] column;

    return true;
}

/*! \fn computeInterpolatorWeights
 * \brief Computes generalized DC-PSE interpolator operators on the set of points.
 * 
 * This function uses one set of points as input data to compute the generalized DC-PSE 
 * interpolator operators on them.
 * 
 * \param idata    A pointer to input data 
 * \param nPoints  Number of data points
 * \param order    Order of accuracy (default is 2nd order accurate)
 * \param nENN     Number of extra nearest neighbors to aid in case of sigularity of the Vandermonde matrix (default is 2)
 * \param ratio    The \f$ \frac{h}{\epsilon} \f$ the default vale is one
 */
template <typename T>
bool dcpse<T>::computeInterpolatorWeights(T *idata, int const nPoints, int order, int nENN, T ratio)
{
    if (nPoints < 1)
    {
        UMUQFAILRETURN("Number of input data points are negative!");
    }

    // Extra check on the order
    order = (order > 0) ? order : 2;
    {
        int *o = this->Order.get();
        std::fill(o, o + this->nTerms, order);
    }

    // Extra check on the number of extra nearest neighbors
    nENN = (nENN > 0) ? nENN : 0;

    // Extra check on the ratio
    ratio = (ratio > 0) ? ratio : static_cast<T>(1);

    // Create an instance of a polynomial object with polynomial degree of \f$ |\beta| + r - 1 \f$
    polynomial<T> poly(this->nDim, order - 1);

    /* 
     * Get the monomials size
     * \f$ monomialSize = \left(\begin{matrix} r + d -1 \\ d \end{matrix}\right) \f$
     */
    this->dcmonomialSize = poly.monomialsize();

    if (nPoints * this->dcmonomialSize > this->dckernelSize)
    {
        this->dckernelSize = nPoints * this->dcmonomialSize;
        try
        {
            // Make sure of the correct kernel size
            this->dckernel.reset(new T[this->dckernelSize]);

            this->h_average.resize(nPoints);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        this->dckernelSize = nPoints * this->dcmonomialSize;
    }

    if (this->KNN)
    {
        if (nPoints != this->KNN->numInputdata() || nPoints != this->KNN->numQuerydata())
        {
            try
            {
                /* 
                 * Finding K nearest neighbors
                 * The number of points K in the neighborhood of each point
                 * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
                 */
                this->KNN.reset(new L2NearestNeighbor<T>(nPoints, nPoints, this->nDim, this->dcmonomialSize + nENN));
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
            this->KNN.reset(new L2NearestNeighbor<T>(nPoints, nPoints, this->nDim, this->dcmonomialSize + nENN));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    // Construct a kd-tree index & do nearest neighbors search
    this->KNN->buildIndex(idata);

    /*
     * Filling the right hand side \f$ b \f$ of the linear system for the kernel coefficients
     * \f$  {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b}  \f$
     */
    EVectorX<T> RHSB0 = EVectorX<T>::Zero(this->dcmonomialSize);
    RHSB0(0) = static_cast<T>(1);
    EVectorX<T> RHSB(this->dcmonomialSize);

    // Total number of nearest neighbours for each point
    int nNN = this->KNN->numNearestNeighbors();

    /*
     * Creating a transpose of the Vandermonde matrix
     * with the size of monomials * monomials \f$  = l \times l \f$
     */
    EMatrixX<T> VMT(this->dcmonomialSize, this->dcmonomialSize);
    EMatrixX<T> VMTimage(this->dcmonomialSize, nNN);

    // Matrix of exponential window function
    EVectorX<T> EM(this->dcmonomialSize);
    EVectorX<T> EMimage(nNN);

    EVectorX<T> columnL(this->dcmonomialSize);

    // Matrix A of a linear system for the kernel coefficients
    EMatrixX<T> AM(this->dcmonomialSize, this->dcmonomialSize);

    // Matrix B^T
    EMatrixX<T> BMT(this->dcmonomialSize, this->dcmonomialSize);
    EMatrixX<T> BMTimage(this->dcmonomialSize, nNN);

    // ${\mathbf a}^T({\mathbf x})$ is the column vector of coefficients which is the solution of linear system
    EVectorX<T> SV(this->dcmonomialSize);

    // Array for keeping the component-wise L1 distances
    std::vector<T> L1Dist(nNN * this->nDim);

    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
    T *column = new T[this->dcmonomialSize];

    // Array to kepp indexing
    std::vector<int> IndexId(this->dcmonomialSize);

    // Loop over all query points
    for (int i = 0; i < nPoints; i++)
    {
        // Index inside kernel
        std::ptrdiff_t const IdM = i * this->dcmonomialSize;

        // Index in idata array
        std::ptrdiff_t const IdI = i * this->nDim;

        // A pointer to nearest neighbors indices of point i
        int *NearestNeighbors = this->KNN->NearestNeighbors(i);

        // A pointer to nearest neighbors distances from point i
        T *nnDist = this->KNN->NearestNeighborsDistances(i);

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
                std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->nDim;

                // pointer to idata (neighbors of i)
                T *Jdata = idata + IdJ;

                for (int d = 0; d < this->nDim; d++, n++)
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

        this->h_average[i] = h_avg;

        // Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
        T const byEpsilon = ratio / h_avg;
        T const byEpsilonsq = byEpsilon * byEpsilon;
        T const byEpsilonsq2 = 0.5 * byEpsilonsq;

        // Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](T &l_i) { l_i *= byEpsilon; });

        // Use the correct RHS for each point
        RHSB = RHSB0;

        // Loop through the neighbors
        for (int j = 0; j < this->dcmonomialSize; j++)
        {
            // Id in the L1 distance list
            std::ptrdiff_t const Id = j * this->nDim;

            // Evaluates a monomial at a point \f$ {\mathbf x} \f$
            poly.monomial_value(L1Dist.data() + Id, column);

            TEMapVectorX<T> columnV(column, this->dcmonomialSize);

            // Fill the Vandermonde matrix column by column
            VMT.block(0, j, this->dcmonomialSize, 1) << columnV;
        }

        for (int j = 0; j < this->dcmonomialSize; j++)
        {
            EM(j) = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq2);
        }

        int dcrank;

        {
            // LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<T>> lu(VMT);

            dcrank = lu.rank();

            if (dcrank < this->dcmonomialSize && dcrank >= this->dcmonomialSize - nENN)
            {
                for (int j = 0; j < this->dcmonomialSize; j++)
                {
                    IndexId[j] = lu.permutationQ().indices()(j);
                }
            }
        }

        if (dcrank < this->dcmonomialSize)
        {
            UMUQWARNING("There are some singularities! we use a least-squares solution!");

            // if necessary, remove redundant equations/coefficients

            // Number of neighbor points are not enough
            if (dcrank < this->dcmonomialSize - nENN)
            {
                UMUQWARNING("Number of neighbor points are not enough! Matrix rank = ");
                std::cerr << dcrank << " < " << this->dcmonomialSize - nENN << std::endl;

                if (nENN > 0)
                {
                    VMTimage.block(0, 0, this->dcmonomialSize, this->dcmonomialSize) << VMT;
                    EMimage.head(this->dcmonomialSize) << EM;

                    // Loop through the rest of nearest neighbors
                    for (int j = this->dcmonomialSize; j < nNN; j++)
                    {
                        // Id in the list
                        std::ptrdiff_t const Id = j * this->nDim;

                        // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                        poly.monomial_value(L1Dist.data() + Id, column);

                        TEMapVectorX<T> columnV(column, this->dcmonomialSize);

                        // Fill the Vandermonde matrix column by column
                        VMTimage.block(0, j, this->dcmonomialSize, 1) << columnV;
                    }

                    for (int j = this->dcmonomialSize; j < nNN; j++)
                    {
                        EMimage(j) = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq2);
                    }

                    /* 
                     * \f[ 
                     * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix} 
                     * \f]
                     */
                    BMTimage = VMTimage * EMatrixX<T>(EMimage.asDiagonal());
                    AM = BMTimage * BMTimage.transpose();
                }
                else
                {
                    /* 
                     * \f[ 
                     * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix} 
                     * \f]
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
                for (int j = dcrank, k = this->dcmonomialSize; j < this->dcmonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t const Id = k * this->nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomial_value(L1Dist.data() + Id, column);

                    TEMapVectorX<T> columnV(column, this->dcmonomialSize);

                    // Get the column l which causes singularity
                    columnL << VMT.block(0, l, this->dcmonomialSize, 1);

                    // Fill the Vandermonde matrix by the new column
                    VMT.block(0, l, this->dcmonomialSize, 1) << columnV;
                }

                for (int j = dcrank, k = this->dcmonomialSize; j < this->dcmonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    EM(l) = std::exp(-nnDist[k] * nnDist[k] * byEpsilonsq2);
                }

                /* 
                 * \f[ 
                 * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                 * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                 * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                 * \end{matrix} 
                 * \f]
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

            // TODO: Correct IndexId in the case of SVD. Right now, this is the best I can do
            /*
             * Later I should check on SVD solution and to find out which columns are the
             * Most important one, then I can correct the IndexId order
             */

            if (dcrank < this->dcmonomialSize - nENN)
            {
                // Loop through the neighbors
                for (int j = 0; j < this->dcmonomialSize; j++)
                {
                    // Id in the list
                    std::ptrdiff_t const Id = j * this->nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomial_value(L1Dist.data() + Id, column);

                    TEMapVectorX<T> columnV(column, this->dcmonomialSize);

                    T const expo = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + j;
                    this->dckernel[IdK] = SV.dot(columnV) * expo;
                }
            }
            else
            {
                // Loop through the neighbors
                for (int j = 0, m = this->dcmonomialSize; j < this->dcmonomialSize; j++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t Id;
                    T expo;

                    if (j >= dcrank)
                    {
                        // Id in the list
                        Id = m * this->nDim;
                        expo = std::exp(-nnDist[m] * nnDist[m] * byEpsilonsq);
                        m++;
                    }
                    else
                    {
                        Id = l * this->nDim;
                        expo = std::exp(-nnDist[l] * nnDist[l] * byEpsilonsq);
                    }

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomial_value(L1Dist.data() + Id, column);

                    TEMapVectorX<T> columnV(column, this->dcmonomialSize);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + l;
                    this->dckernel[IdK] = SV.dot(columnV) * expo;
                }

                // Loop through the neighbors
                for (int j = dcrank, m = this->dcmonomialSize; j < this->dcmonomialSize; j++, m++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Correct the neighborhood order
                    this->KNN->IndexSwap(l, m);
                }
            }
        }
        else
        {
            /* 
             * \f[ 
             * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
             * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
             * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
             * \end{matrix} 
             * \f]
             */
            BMT = VMT * EMatrixX<T>(EM.asDiagonal());

            AM = BMT * BMT.transpose();

            // SV contains the solution of \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b} \f$
            SV = AM.lu().solve(RHSB);

            // Loop through the neighbors
            for (int j = 0; j < this->dcmonomialSize; j++)
            {
                // Id in the list
                std::ptrdiff_t const Id = j * this->nDim;

                // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                poly.monomial_value(L1Dist.data() + Id, column);

                TEMapVectorX<T> columnV(column, this->dcmonomialSize);

                T const expo = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq);

                // Index inside the kernel
                std::ptrdiff_t const IdK = IdM + j;
                this->dckernel[IdK] = SV.dot(columnV) * expo;
            }
        }

    } // Loop over all points

    delete[] column;

    return true;
}

/*! \fn computeInterpolatorWeights
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
 * \param nENN      Number of extra nearest neighbors to aid in case of sigularity of the Vandermonde matrix (default is 2)
 * \param ratio     The \f$ \frac{h}{\epsilon} \f$ the default vale is one
 */
template <typename T>
bool dcpse<T>::computeInterpolatorWeights(T *idata, int const nPoints, T *qdata, int const nqPoints, int order, int nENN, T ratio)
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
        int *o = this->Order.get();
        std::fill(o, o + this->nTerms, order);
    }

    // Extra check on the number of extra nearest neighbors
    nENN = (nENN > 0) ? nENN : 0;

    // Extra check on the ratio
    ratio = (ratio > 0) ? ratio : static_cast<T>(1);

    // Create an instance of a polynomial object with polynomial degree of \f$ |\beta| + r - 1 \f$
    polynomial<T> poly(this->nDim, order - 1);

    /* 
     * Get the monomials size
     * \f$ monomialSize = \left(\begin{matrix} r + d -1 \\ d \end{matrix}\right) \f$
     */
    this->dcmonomialSize = poly.monomialsize();

    if (nqPoints * this->dcmonomialSize > this->dckernelSize)
    {
        this->dckernelSize = nqPoints * this->dcmonomialSize;
        try
        {
            // Make sure of the correct kernel size
            this->dckernel.reset(new T[this->dckernelSize]);
            this->h_average.resize(nqPoints);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        this->dckernelSize = nqPoints * this->dcmonomialSize;
    }

    if (this->KNN)
    {
        if (nqPoints != this->KNN->numInputdata() || nqPoints != this->KNN->numQuerydata())
        {
            try
            {
                /* 
                 * Finding K nearest neighbors
                 * The number of points K in the neighborhood of each point
                 * \f$ K = \text{monomial size} + \text{number of extra neighbors} \f$
                 */
                this->KNN.reset(new L2NearestNeighbor<T>(nPoints, nqPoints, this->nDim, this->dcmonomialSize + nENN));
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
            this->KNN.reset(new L2NearestNeighbor<T>(nPoints, nqPoints, this->nDim, this->dcmonomialSize + nENN));
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    // Construct a kd-tree index & do nearest neighbors search
    this->KNN->buildIndex(idata, qdata);

    // Vector of all idata points' distances from their closest nearest neighbor
    T *idataminDist = nullptr;

    {
        // Finding only one nearest neighbor for the input data points
        L2NearestNeighbor<T> KNN1(nPoints, this->nDim, 1);

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
    EVectorX<T> RHSB0 = EVectorX<T>::Zero(this->dcmonomialSize);
    RHSB0(0) = static_cast<T>(1);
    EVectorX<T> RHSB(this->dcmonomialSize);

    // Total number of nearest neighbours for each point
    int nNN = this->KNN->numNearestNeighbors();

    /*
     * Creating a transpose of the Vandermonde matrix
     * with the size of monomials * monomials \f$  = l \times l \f$
     */
    EMatrixX<T> VMT(this->dcmonomialSize, this->dcmonomialSize);
    EMatrixX<T> VMTimage(this->dcmonomialSize, nNN);

    // Matrix of exponential window function
    EVectorX<T> EM(this->dcmonomialSize);
    EVectorX<T> EMimage(nNN);

    EVectorX<T> columnL(this->dcmonomialSize);

    // Matrix A of a linear system for the kernel coefficients
    EMatrixX<T> AM(this->dcmonomialSize, this->dcmonomialSize);

    // Matrix B^T
    EMatrixX<T> BMT(this->dcmonomialSize, this->dcmonomialSize);
    EMatrixX<T> BMTimage(this->dcmonomialSize, nNN);

    // ${\mathbf a}^T({\mathbf x})$ is the column vector of coefficients which is the solution of linear system
    EVectorX<T> SV(this->dcmonomialSize);

    // Array for keeping the component-wise L1 distances
    std::vector<T> L1Dist(nNN * this->nDim);

    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
    T *column = new T[this->dcmonomialSize];

    // Array to kepp indexing
    std::vector<int> IndexId(this->dcmonomialSize);

    // Primitive (quartic spline) object
    quartic_spline<T> q;

    // Loop over all query points
    for (int i = 0; i < nqPoints; i++)
    {
        // Index inside kernel
        std::ptrdiff_t const IdM = i * this->dcmonomialSize;

        // Index in qdata array
        std::ptrdiff_t const IdI = i * this->nDim;

        // A pointer to nearest neighbors indices of point i
        int *NearestNeighbors = this->KNN->NearestNeighbors(i);

        // A pointer to nearest neighbors distances from point i
        T *nnDist = this->KNN->NearestNeighborsDistances(i);

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
                std::ptrdiff_t const IdJ = NearestNeighbors[j] * this->nDim;

                // pointer to idata (neighbors of i)
                T *Jdata = idata + IdJ;

                for (int d = 0; d < this->nDim; d++, n++)
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

        this->h_average[i] = h_avg;

        // Computing the smoothing length for each point \f$ \frac{h}{\epsilon} \sim ratio \f$
        T const byEpsilon = ratio / h_avg;
        T const byEpsilonsq = byEpsilon * byEpsilon;
        T const byEpsilonsq2 = 0.5 * byEpsilonsq;

        // Vectors pointing to \f$ {\mathbf x} \f$ from all neighboring points
        std::for_each(L1Dist.begin(), L1Dist.end(), [&](T &l_i) { l_i *= byEpsilon; });

        // Use the correct RHS for each point
        RHSB = RHSB0;

        // Loop through the neighbors
        for (int j = 0; j < this->dcmonomialSize; j++)
        {
            // Id in the L1 distance list
            std::ptrdiff_t const Id = j * this->nDim;

            // Evaluates a monomial at a point \f$ {\mathbf x} \f$
            poly.monomial_value(L1Dist.data() + Id, column);

            TEMapVectorX<T> columnV(column, this->dcmonomialSize);

            // Fill the Vandermonde matrix column by column
            VMT.block(0, j, this->dcmonomialSize, 1) << columnV;

            // Neighbor point number
            int const IdJ = NearestNeighbors[j];

            /* 
             * Using a smooth correction function that satisfies
             * \f$ {\mathbf F} \left(\frac{{\mathbf x}_p-{\mathbf x}_q}{c({\mathbf x}_q)} \right) =\delta_{pq} \f$
             * Choose \f$ c({\mathbf x}) \f$ such that it is smaller than the distance
             * between the point and its nearest neighbors
             */
            T s = nnDist[j] / (0.9 * idataminDist[IdJ]);

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
            this->dckernel[IdK] = dckernelV;
        }

        for (int j = 0; j < this->dcmonomialSize; j++)
        {
            EM(j) = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq2);
        }

        int dcrank;

        {
            // LU decomposition of a matrix with complete pivoting, and related features.
            Eigen::FullPivLU<EMatrixX<T>> lu(VMT);

            dcrank = lu.rank();

            if (dcrank < this->dcmonomialSize && dcrank >= this->dcmonomialSize - nENN)
            {
                for (int j = 0; j < this->dcmonomialSize; j++)
                {
                    IndexId[j] = lu.permutationQ().indices()(j);
                }
            }
        }

        if (dcrank < this->dcmonomialSize)
        {
            UMUQWARNING("There are some singularities! we use a least-squares solution!");

            // if necessary, remove redundant equations/coefficients

            // Number of neighbor points are not enough
            if (dcrank < this->dcmonomialSize - nENN)
            {
                UMUQWARNING("Number of neighbor points are not enough! Matrix rank = ");
                std::cerr << dcrank << " < " << this->dcmonomialSize - nENN << std::endl;

                if (nENN > 0)
                {
                    VMTimage.block(0, 0, this->dcmonomialSize, this->dcmonomialSize) << VMT;
                    EMimage.head(this->dcmonomialSize) << EM;

                    // Loop through the rest of nearest neighbors
                    for (int j = this->dcmonomialSize; j < nNN; j++)
                    {
                        // Id in the list
                        std::ptrdiff_t const Id = j * this->nDim;

                        // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                        poly.monomial_value(L1Dist.data() + Id, column);

                        TEMapVectorX<T> columnV(column, this->dcmonomialSize);

                        // Fill the Vandermonde matrix column by column
                        VMTimage.block(0, j, this->dcmonomialSize, 1) << columnV;

                        // Neighbor point number
                        int const IdJ = NearestNeighbors[j];

                        /*
                         * Using a smooth correction function that satisfies
                         * \f$ {\mathbf F} \left(\frac{{\mathbf x}_p-{\mathbf x}_q}{c({\mathbf x}_q)} \right) =\delta_{pq} \f$
                         * Choose \f$ c({\mathbf x}) \f$ such that it is smaller than the distance
                         * between the point and its nearest neighbors
                         */
                        T s = nnDist[j] / (0.9 * idataminDist[IdJ]);

                        // Compute the kernel value at the point
                        T dckernelV = q.f(&s);

                        /*
                         * Assemble the right hand side
                         * \f[
                         *  {\mathbf b}={\mathbf P}({\mathbf x}) |_{{\mathbf x}=0} - 
                         * \sum_{p} {\mathbf P}{\left(\frac{{\mathbf x}-{\mathbf x}_p}{\epsilon({\mathbf x})}\right)} 
                         * {\mathbf C}\left(\frac{{\mathbf x}-{\mathbf x}_p}{c({\mathbf x}_p)} \right) 
                         * \f]
                         */
                        RHSB -= dckernelV * columnV;
                    }

                    for (int j = this->dcmonomialSize; j < nNN; j++)
                    {
                        EMimage(j) = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq2);
                    }

                    /* 
                     * \f[ 
                     * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix} 
                     * \f]
                     */
                    BMTimage = VMTimage * EMatrixX<T>(EMimage.asDiagonal());
                    AM = BMTimage * BMTimage.transpose();
                }
                else
                {
                    /* 
                     * \f[ 
                     * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                     * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                     * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                     * \end{matrix} 
                     * \f]
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
                for (int j = dcrank, k = this->dcmonomialSize; j < this->dcmonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t const Id = k * this->nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomial_value(L1Dist.data() + Id, column);

                    TEMapVectorX<T> columnV(column, this->dcmonomialSize);

                    // Get the column l which causes singularity
                    columnL << VMT.block(0, l, this->dcmonomialSize, 1);

                    // Fill the Vandermonde matrix by the new column
                    VMT.block(0, l, this->dcmonomialSize, 1) << columnV;

                    // Neighbor point number
                    int const IdJ = NearestNeighbors[k];

                    /* 
                     * Using a smooth correction function that satisfies
                     * \f$ {\mathbf F} \left(\frac{{\mathbf x}_p-{\mathbf x}_q}{c({\mathbf x}_q)} \right) =\delta_{pq} \f$
                     * Choose \f$ c({\mathbf x}) \f$ such that it is smaller than the distance
                     * between the point and its nearest neighbors
                     */
                    T s = nnDist[k] / (0.9 * idataminDist[IdJ]);

                    // Compute the kernel value at the point IdK
                    T dckernelV = q.f(&s);

                    // Index of the column l inside the kernel
                    std::ptrdiff_t const IdK = IdM + j;

                    this->dckernel[IdK] = dckernelV;

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
                    s = nnDist[l] / (0.9 * idataminDist[IdJL]);
                    dckernelV = q.f(&s);
                    RHSB += dckernelV * columnL;
                }

                for (int j = dcrank, k = this->dcmonomialSize; j < this->dcmonomialSize; j++, k++)
                {
                    // Get the column number which causes a singularity
                    int const l = IndexId[j];

                    EM(l) = std::exp(-nnDist[k] * nnDist[k] * byEpsilonsq2);
                }

                /* 
                 * \f[ 
                 * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
                 * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
                 * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
                 * \end{matrix} 
                 * \f]
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

            // TODO: Correct IndexId in the case of SVD. Right now, this is the best I can do
            /*
             * Later I should check on SVD solution and to find out which columns are the
             * Most important one, then I can correct the IndexId order
             */

            if (dcrank < this->dcmonomialSize - nENN)
            {
                // Loop through the neighbors
                for (int j = 0; j < this->dcmonomialSize; j++)
                {
                    // Id in the list
                    std::ptrdiff_t const Id = j * this->nDim;

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomial_value(L1Dist.data() + Id, column);

                    TEMapVectorX<T> columnV(column, this->dcmonomialSize);

                    T const expo = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + j;
                    this->dckernel[IdK] += SV.dot(columnV) * expo;
                }
            }
            else
            {
                // Loop through the neighbors
                for (int j = 0, m = this->dcmonomialSize; j < this->dcmonomialSize; j++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Id in the list
                    std::ptrdiff_t Id;
                    T expo;

                    if (j >= dcrank)
                    {
                        // Id in the list
                        Id = m * this->nDim;
                        expo = std::exp(-nnDist[m] * nnDist[m] * byEpsilonsq);
                        m++;
                    }
                    else
                    {
                        Id = l * this->nDim;
                        expo = std::exp(-nnDist[l] * nnDist[l] * byEpsilonsq);
                    }

                    // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                    poly.monomial_value(L1Dist.data() + Id, column);

                    TEMapVectorX<T> columnV(column, this->dcmonomialSize);

                    // Index inside the kernel
                    std::ptrdiff_t const IdK = IdM + l;
                    this->dckernel[IdK] += SV.dot(columnV) * expo;
                }

                // Loop through the neighbors
                for (int j = dcrank, m = this->dcmonomialSize; j < this->dcmonomialSize; j++, m++)
                {
                    // Get the right index
                    int const l = IndexId[j];

                    // Correct the neighborhood order
                    this->KNN->IndexSwap(l, m);
                }
            }
        }
        else
        {
            /* 
             * \f[ 
             * \begin{matrix} {\mathbf A} ({\mathbf x}) = {\mathbf B}^T ({\mathbf x}) {\mathbf B} ({\mathbf x}) & \in \mathbb{R}^{l\times l} \\
             * {\mathbf B} ({\mathbf x}) = {\mathbf E} ({\mathbf x}) {\mathbf V} ({\mathbf x}) & \in \mathbb{R}^{k\times l}\\
             * {\mathbf b} = (-1)^{|\beta|} D^\beta {\mathbf P}({\mathbf x}) |_{{\mathbf x}=0}   & \in \mathbb{R}^{l\times 1}
             * \end{matrix} 
             * \f]
             */
            BMT = VMT * EMatrixX<T>(EM.asDiagonal());

            AM = BMT * BMT.transpose();

            // SV contains the solution of \f$ {\mathbf A} ({\mathbf x}) {\mathbf a}^T({\mathbf x})={\mathbf b} \f$
            SV = AM.lu().solve(RHSB);

            // Loop through the neighbors
            for (int j = 0; j < this->dcmonomialSize; j++)
            {
                // Id in the list
                std::ptrdiff_t const Id = j * this->nDim;

                // Evaluates a monomial at a point \f$ {\mathbf x} \f$
                poly.monomial_value(L1Dist.data() + Id, column);

                TEMapVectorX<T> columnV(column, this->dcmonomialSize);

                T const expo = std::exp(-nnDist[j] * nnDist[j] * byEpsilonsq);

                // Index inside the kernel
                std::ptrdiff_t const IdK = IdM + j;
                this->dckernel[IdK] += SV.dot(columnV) * expo;
            }
        }

    } // Loop over all points

    delete[] column;
    delete[] idataminDist;

    return true;
}

/*! \fn compute
 * \brief Evaluate a discretized DC-PSE operator from function values of input data and put the results as the query data function values
 * 
 * This function uses function values of input data and the weights of the operator which have 
 * been previously computed to compute the query values and put the results as the query data 
 * function values. 
 * 
 * At first it checks the computed kernel size to be equal to the number of query points times the 
 * size of monomials which has been previously computed for the required degree of the DC-PSE operator.
 * 
 * \param iFvalue   A pointer to input data function value
 * \param nPoints   Number of data points
 * \param qFvalue   A pointer to query data function value
 * \param nqPoints  Number of query data points
 */
template <typename T>
bool dcpse<T>::compute(T *iFvalue, int const nPoints, T *qFvalue, int const nqPoints)
{
    if (this->KNN->numInputdata() != nPoints)
    {
        UMUQFAILRETURN("Input data does not match with previously computed weights!");
    }
    if (this->KNN->numQuerydata() != nqPoints)
    {
        UMUQFAILRETURN("Query data does not match with previously computed weights!");
    }
    if (this->dckernelSize != nqPoints * this->dcmonomialSize)
    {
        UMUQFAILRETURN("Previously computed weights does not macth with this query data!");
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

/*! \fn interpolate
 * \brief Evaluate a discretized DC-PSE interpolation operator from function values of input data and put the 
 * interpolation results as the query data values
 * 
 * This function uses function values of input data and the weights of the interpolation operator which have 
 * been previously computed to compute the query values and put the results as the query data 
 * function values. 
 * 
 * At first it checks the computed kernel size to be equal to the number of query points times the 
 * size of monomials which has been previously computed for the required degree of DC-PSE operator
 * or interpolator.
 * 
 * \param iFvalue   A pointer to input data function value
 * \param nPoints   Number of data points
 * \param qFvalue   A pointer to query data function value
 * \param nqPoints  Number of query data points
 */
template <typename T>
bool dcpse<T>::interpolate(T *iFvalue, int const nPoints, T *qFvalue, int const nqPoints)
{
    if (this->KNN->numInputdata() != nPoints)
    {
        UMUQFAILRETURN("Input data does not match with previously computed weights!");
    }
    if (this->KNN->numQuerydata() != nqPoints)
    {
        UMUQFAILRETURN("Query data does not match with previously computed weights!");
    }
    if (this->dckernelSize != nqPoints * this->dcmonomialSize)
    {
        UMUQFAILRETURN("Previously computed weights does not macth with this query data!");
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
        int *NearestNeighbors = this->KNN->NearestNeighbors(i);

        int IdI = i * this->dcmonomialSize;

        T sum(0);

        std::cout << "For point i=" << i << " dckernel=";
        // Loop through the neighbors
        for (int j = 0; j < this->dcmonomialSize; j++, IdI++)
        {
            int const IdJ = NearestNeighbors[j];
            sum += this->dckernel[IdI] * iFvalue[IdJ];

            std::cout << this->dckernel[IdI] << " ";
        }
        std::cout << "Fvalue=" << sum << " h_average=" << this->h_average[i] << std::endl;
        qFvalue[i] = sum;
    }

    return true;
}

/*!
 * \brief A pointer to neighborhood kernel at index
 * 
 * \param index Index of a point (from data points) to get its neighborhood kernel
 * 
 * \returns A (pointer to a) row of the nearest neighbors kernel values.
 */
template <typename T>
inline T *dcpse<T>::neighborhoodKernel(int const index) const
{
    return this->dckernel.get() + index * this->dcmonomialSize;
}

/*!
 * \brief A pointer to kernel array of all query points
 * 
 * \returns  A pointer to kernel array of all query points
 */
template <typename T>
inline T *dcpse<T>::neighborhoodKernel() const
{
    return this->dckernel.get();
}

/*!
 * \brief Size of the neighborhood kernel which equals to the monomial size 
 * 
 * \returns Size of the neighborhood kernel
 */
template <typename T>
inline int dcpse<T>::neighborhoodKernelSize() const
{
    return this->dcmonomialSize;
}

/*!
 * \brief Order of accuracy of DC-PSE kernel at index
 * 
 * \param  index Index number in nTerms array
 * \return order of accuracy of DC-PSE kernel at index
 */
template <typename T>
inline int dcpse<T>::orderofAccuracy(int const index) const
{
    return this->Order[index];
}

/*!
 * \brief Print the DC-PSE information
 * 
 */
template <typename T>
inline void dcpse<T>::printInfo() const
{
    for (int i = 0; i < this->nTerms; i++)
    {
        std::cout << this->Order[i] << (this->Order[i] % 10 == 1 ? "st " : this->Order[i] % 10 == 2 ? "nd " : this->Order[i] % 10 == 3 ? "rd " : "th ") << "order DC-PSE kernel uses \n"
                  << this->neighborhoodKernelSize() << " points in the neighborhood of each query points." << std::endl;
    }
}

/*!
 * \brief Component-wise average neighbor spacing @index
 * 
 * \param index Index of a point (from data points) to get its average neighbor spacing
 * 
 * \returns A component-wise average neighbor spacing @index 
 */
template <typename T>
inline T dcpse<T>::averageSpace(int const index) const
{
    return this->h_average[index];
}

/*!
 * \brief A pointer to component-wise average neighbor spacing
 * 
 * \returns A pointer to component-wise average neighbor spacing
 */
template <typename T>
inline T *dcpse<T>::averageSpace() const
{
    return this->h_average.data();
}

#endif

#ifndef UMUQ_LINEARREGRESSION_H
#define UMUQ_LINEARREGRESSION_H

#include "core/core.hpp"
#include "polynomials.hpp"
#include "eigenlib.hpp"

#include <cstddef>

#include <vector>
#include <utility>
#include <memory>
#include <new>
#include <algorithm>

namespace umuq
{

/*! \class linearRegression
 * \ingroup Numerics_Module
 *
 * \brief linearRegression is a linear approach to modelling the relationship between a
 * scalar response (or dependent variable) and one or more explanatory variables (or
 * independent variables).
 *
 * \tparam DataType        Data type
 * \tparam PolynomialType  Polynomial type (Default is - \b polynomial<DataType> with monomials)<br>
 *                         \sa umuq::polynomials::PolynomialTypes.<br>
 */
template <typename DataType, class PolynomialType = polynomial<DataType>>
class linearRegression
{
  public:
    /*!
     * \brief Construct a new linear Regression object
     *
     * \param ndim             Number of dimensions
     * \param polynomialorder  Polynomial order (default 1 which is a n-dimensional linear polynomial)
     */
    linearRegression(int ndim, int polynomialorder = 1);

    /*!
     * \brief Move constructor, construct a new linearRegression object from input linearRegression object
     *
     * \param other linearRegression object
     */
    linearRegression(linearRegression<DataType, PolynomialType> &&other);

    /*!
     * \brief Move assignment operator
     *
     * \param other  linearRegression object
     *
     * \return linearRegression<DataType, PolynomialType>&
     */
    linearRegression<DataType, PolynomialType> &operator=(linearRegression<DataType, PolynomialType> &&other);

    /*!
     * \brief Destroy the linearRegression object
     *
     */
    ~linearRegression();

    /*!
     * \brief Compute the linear regression kernel weights
     *
     * \param idata    Input data points
     * \param iFvalue  Function value at the input points
     * \param nPoints  Number of input points
     *
     * \return false   For wrong number of input points or not having enough memory
     */
    bool computeWeights(DataType *idata, DataType *iFvalue, int const nPoints);

    /*!
     * \brief Solution for the new points using the computed Kernel weights
     *
     * \param qdata     N-dimensional input query data points
     * \param qFvalue   [out] Value of the function at the query points
     * \param nqPoints  Number of query points
     *
     * \return false    If polynomialOrder has been changed between computing the kernels and solution
     *                  or not having enough memory
     */
    bool solve(DataType *qdata, DataType *qFvalue, int const nqPoints);

    /*!
     * \brief Minimum number of points which is required to do the linear regression
     *
     * \return Minimum number of points
     */
    inline int minPointsRequired();

    /*!
     * \brief Minimum number of points which is required to do the linear regression
     *
     * \return Recommended number of points
     */
    inline int recommendedNumPoints();

    /*!
     * \brief Set the new Polynomial Order object
     *
     * \param polynomialorder new polynomial order
     */
    void resetPolynomialOrder(int const polynomialorder);

  private:
    /*!
     * \brief Delete a linearRegression object copy construction
     *
     * Avoiding implicit generation of the copy constructor.
     */
    linearRegression(linearRegression<DataType, PolynomialType> const &) = delete;

    /*!
     * \brief Delete a linearRegression object assignment
     *
     * Avoiding implicit copy assignment.
     */
    linearRegression<DataType, PolynomialType> &operator=(linearRegression<DataType, PolynomialType> const &) = delete;

  private:
    //! Dimensionality
    int nDim;

    //! Polynomial order
    int polynomialOrder;

    //! The monomial size
    int linearRegressionMonomialSize;

    //! Size of the kernel
    int linearRegressionkernelSize;

    //! Operator kernel
    std::vector<DataType> linearRegressionkernel;
};

template <typename DataType, class PolynomialType>
linearRegression<DataType, PolynomialType>::linearRegression(int ndim, int polynomialorder) : nDim(ndim),
                                                                                              polynomialOrder(polynomialorder),
                                                                                              linearRegressionMonomialSize(0),
                                                                                              linearRegressionkernelSize(0) {}

template <typename DataType, class PolynomialType>
linearRegression<DataType, PolynomialType>::linearRegression(linearRegression<DataType, PolynomialType> &&other)
{
    nDim = other.nDim;
    polynomialOrder = other.polynomialOrder;
    linearRegressionMonomialSize = other.linearRegressionMonomialSize;
    linearRegressionkernelSize = other.linearRegressionkernelSize;
    linearRegressionkernel = std::move(other.linearRegressionkernel);
}

template <typename DataType, class PolynomialType>
linearRegression<DataType, PolynomialType> &linearRegression<DataType, PolynomialType>::operator=(linearRegression<DataType, PolynomialType> &&other)
{
    nDim = other.nDim;
    polynomialOrder = other.polynomialOrder;
    linearRegressionMonomialSize = other.linearRegressionMonomialSize;
    linearRegressionkernelSize = other.linearRegressionkernelSize;
    linearRegressionkernel = std::move(other.linearRegressionkernel);

    return *this;
}

template <typename DataType, class PolynomialType>
linearRegression<DataType, PolynomialType>::~linearRegression() {}

template <typename DataType, class PolynomialType>
bool linearRegression<DataType, PolynomialType>::computeWeights(DataType *idata, DataType *iFvalue, int const nPoints)
{
    if (nPoints < 1 || nPoints < minPointsRequired())
    {
        UMUQFAILRETURN("Number of input data points =", nPoints, " are not enough!");
    }

    // Create an instance of a polynomial object with polynomial order
    PolynomialType poly(nDim, polynomialOrder);

    // Get the monomials size
    linearRegressionMonomialSize = poly.monomialsize();

    if (linearRegressionMonomialSize > linearRegressionkernelSize)
    {
        linearRegressionkernelSize = linearRegressionMonomialSize;
        try
        {
            // Make sure of the correct kernel size
            linearRegressionkernel.resize(linearRegressionkernelSize);
        }
        catch (...)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }
    else
    {
        linearRegressionkernelSize = linearRegressionMonomialSize;
    }

    // Right hand side vector
    EVectorMapType<DataType> BV(iFvalue, nPoints);

    // Matrix A
    EMatrixX<DataType> AM(nPoints, linearRegressionMonomialSize);

    // Solution vector
    DataType *sv = linearRegressionkernel.data();
    EVectorMapType<DataType> SV(sv, linearRegressionMonomialSize);

    // dummy array of data
    std::unique_ptr<DataType[]> rowData(new DataType[linearRegressionMonomialSize]);
    DataType *rowdata = rowData.get();

    // Loop over all query points
    for (int i = 0; i < nPoints; i++)
    {
        // Index in idata array
        std::ptrdiff_t const IdI = i * nDim;

        // Evaluates a monomial at a point \f$ {\mathbf x} \f$
        poly.monomialValue(idata + IdI, rowdata);

        // Loop through the neighbors
        for (int j = 0; j < linearRegressionMonomialSize; j++)
        {
            AM(i, j) = rowData[j];
        }

    } // Loop over all points

    {
        /*
         * Two-sided Jacobi SVD decomposition, ensuring optimal reliability and accuracy.
         * Thin U and V are all we need for (least squares) solving.
         */
        Eigen::JacobiSVD<EMatrixX<DataType>> svd(AM, Eigen::DecompositionOptions::ComputeThinU | Eigen::DecompositionOptions::ComputeThinV);

        /*
         * SV contains the least-squares solution of
         * \f$ {\mathbf A} ({\mathbf X}) ={\mathbf b}, \f$
         * using the current SVD decomposition of A.
         */
        SV = svd.solve(BV);
    }

    return true;
}

template <typename DataType, class PolynomialType>
bool linearRegression<DataType, PolynomialType>::solve(DataType *qdata, DataType *qFvalue, int const nqPoints)
{
    // Create an instance of a polynomial object with polynomial order
    PolynomialType poly(nDim, polynomialOrder);

    if (poly.monomialsize() != linearRegressionkernelSize)
    {
        UMUQFAILRETURN("Polynomial order has changed between Linear regression construction and its solution!");
    }

    if (qFvalue == nullptr)
    {
        try
        {
            qFvalue = new DataType[nqPoints];
        }
        catch (std::bad_alloc &e)
        {
            UMUQFAILRETURN("Failed to allocate memory!");
        }
    }

    // Get the pointer to the kernel weights
    DataType *sv = linearRegressionkernel.data();

    // dummy array of data
    std::unique_ptr<DataType[]> rowData(new DataType[linearRegressionMonomialSize]);
    DataType *rowdata;

    // Loop over all query points
    for (int i = 0; i < nqPoints; i++)
    {
        // Index in qdata array
        std::ptrdiff_t const IdI = i * nDim;

        rowdata = rowData.get();

        // Evaluates a monomial at a point \f$ {\mathbf x} \f$
        poly.monomialValue(qdata + IdI, rowdata);

        DataType sum(0);
        std::for_each(sv, sv + linearRegressionMonomialSize, [&](DataType const s) { sum += s * (*rowdata++); });

        qFvalue[i] = sum;
    }

    return true;
}

template <typename DataType, class PolynomialType>
inline int linearRegression<DataType, PolynomialType>::minPointsRequired()
{
    // Create an instance of a polynomial object with polynomial order
    PolynomialType poly(nDim, polynomialOrder);
    /* Get the monomials size \f$ monomialSize = \left(\begin{matrix} r + d -1 \\ d \end{matrix}\right) \f$ */
    return poly.monomialsize();
}

template <typename DataType, class PolynomialType>
inline int linearRegression<DataType, PolynomialType>::recommendedNumPoints()
{
    return minPointsRequired();
}

template <typename DataType, class PolynomialType>
void linearRegression<DataType, PolynomialType>::resetPolynomialOrder(int const polynomialorder)
{
    polynomialOrder = polynomialorder;
    //
    linearRegressionMonomialSize = 0;
    linearRegressionkernelSize = 0;
    linearRegressionkernel.clear();
    linearRegressionkernel.shrink_to_fit();
}

} // namespace umuq

#endif // UMUQ_LINEARREGRESSION

#ifndef UMUQ_RADIALBASISFUNCTION_H
#define UMUQ_RADIALBASISFUNCTION_H

#include "global.hpp"
#include "surrogate.hpp"
#include "radialbasisfunctionkernel.hpp"
#include "polynomialtail.hpp"

namespace umuq
{

/*! \file radialbasisfunction.hpp
 * \ingroup 
 * 
 * \brief Implementation of the Radial basis function interpolant.
 *
 * \author David Eriksson, dme65@cornell.edu
 * 
 * This file contains minor addition to the original rbf.h
 * source code made available under the following license:
 *
 * \verbatim
 * Copyright (c) 2016 by David Eriksson.
 * 
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * \endverbatim
 */

#ifdef DEBUG
extern umuqTimer gTimer;
#endif

/*! \class radialBasisFunction
 * \ingroup Surrogate_Module
 * 
 * \brief Radial basis function
 * 
 * \tparam kernelType         Radial basis function kernel type (Cubic is default)
 * \tparam polynomialTailType Polynomial tail type (Linear is default)
 *
 * A radial basis function (RBF) interpolant is a weighted sum of radial 
 * basis functions. It is common to add a polynomial tail as well to
 * assure that the interpolant can exactly reproduce polynomial of that degree.
 * This leads to an interpolant of the form
 * 
 * \f$ 
 * s(y) = \displaystyle\sum_{i=1}^n \lambda_i \varphi(\|y-x_i\|) + 
 *        \displaystyle\sum_{i=1}^m c_i \pi_i(y)
 * \f$
 * 
 * where \f$ y_i\f$ are the n centers and \f$ \{\pi_i\}_{i=1}^m\f$ is a basis
 * of the polynomial space of the tail. Given a set of points 
 * \f$ X=\{x_1,\ldots,x_n\}\f$ with values \f$ f_X = \{f(x_1),\ldots,f(x_n)\}\f$
 * the interpolation conditions are:<br>
 * 
 * \f$ s(x_i) = f(x_i)\f$ for \f$ i=1,\ldots,n\f$
 * 
 * and in order to get a unique interpolant one usually adds the conditions
 * 
 * \f$ \displaystyle\sum_{i=1}^n \lambda_i \pi_j(x_i) = 0\f$ for \f$ j=1,\ldots,m\f$
 * 
 * which leads to a system of equations of size \f$(m+n) x (m+n)\f$. With the 
 * notation \f$ \Phi_{i,j}=\varphi(\|x_i-x_j\|)\f$ and \f$ P_{ij} = \pi_j(x_i)\f$
 * we can write in a more compact form
 * 
 * \f$ \left(\begin{array}{cc} 0 & P^T \\ P & \Phi \end{array}\right)
 *     \left(\begin{array}{c} c \\ \lambda \end{array}\right) = 
 *     \left(\begin{array}{c} 0 \\ f_X \end{array}\right)
 * \f$
 * 
 * We can see that adding one more point corresponds to adding a column and
 * a row to this matrix which is why this ordering of \f$\Phi\f$ and \f$P\f$
 * is convenient. We store the LU decomposition of this matrix and use the
 * fact that the Schur complement is symmetric and positive definite so we
 * can update the LU-decomposition by computing the Cholesky decomposition
 * of the Schur complement. We can hence add \f$k\f$ new points using
 * roughly \f$2n^2k\f$ flops under the assumption that \f$n\gg k\f$. Solving
 * for the new coefficients is then a matter of back and forward subsitution
 * which will take roughly \f$2(m+n)^2\f$ flops. This is better than solving
 * the system from scratch which takes roughly \f$(2/3)(m+n)^3\f$ flops. 
 * 
 * In order to compute the initial decomposition we need at least m initial points 
 * that serve as "outer" points that are needed to uniquely fit the polynomial
 * tail and to construct the initial LU-decomposition with pivoting. No pivoting
 * is needed from that point.
 * 
 * The domain is automatically scaled to the unit box to avoid scaling
 * issues since the kernel and polynomial tail scale differently.
 * 
 * \author David Eriksson, dme65@cornell.edu
 */
template <class kernelType = cubicKernel, class polynomialTailType = linearPolynomialTail>
class radialBasisFunction : public surrogate
{
  public:
    /*!
     * \brief Construct a new radialBasisFunction object
     * 
     * \param NumDimensions       Number of dimensions
     * \param MaxNumPoints        Capacity Maximum number of possible points
     * \param DampingCoefficient  Damping coefficient (non-negative)
     */
    radialBasisFunction(int const NumDimensions, int const MaxNumPoints, double const DampingCoefficient = 1e-8);

    /*!
     * \brief Construct a new radialBasisFunction object
     * 
     * \param NumDimensions       Number of dimensions
     * \param MaxNumPoints        Capacity Maximum number of possible points
     * \param LowerBounds         Lower variable bounds
     * \param UpperBounds         Upper variable bounds
     * \param DampingCoefficient  Damping coefficient (non-negative)
     */
    radialBasisFunction(int const NumDimensions, int const MaxNumPoints, EVectorXd const &LowerBounds, EVectorXd const &UpperBounds, double const DampingCoefficient = 1e-8);

    /*!
     * \brief Move constructor, construct a new radialBasisFunction object
     * 
     * \param other radialBasisFunction object
     */
    explicit radialBasisFunction(radialBasisFunction<kernelType, polynomialTailType> &&other);

    /*!
     * \brief Move assignment operator
     * 
     * \param other radialBasisFunction object
     * 
     * \returns radialBasisFunction& radialBasisFunction object
     */
    radialBasisFunction<kernelType, polynomialTailType> &operator=(radialBasisFunction<kernelType, polynomialTailType> &&other);

    /*!
     * \brief Destroy the radialBasisFunction object
     * 
     */
    ~radialBasisFunction();

    /*!
     * \brief Method for resetting the surrogate model
     * 
     */
    inline void reset() override;

    /*!
     * \brief Method for resetting the surrogate model
     * 
     * \param NumDimensions       Number of dimensions
     * \param MaxNumPoints        Capacity Maximum number of possible points
     * \param DampingCoefficient  Damping coefficient (non-negative)
     */
    inline void reset(int const NumDimensions, int const MaxNumPoints, double const DampingCoefficient = 1e-8);

    /*!
     * \brief Method for resetting the surrogate model
     * 
     * \param NumDimensions       Number of dimensions
     * \param MaxNumPoints        Capacity Maximum number of possible points
     * \param LowerBounds         Lower variable bounds
     * \param UpperBounds         Upper variable bounds
     * \param DampingCoefficient  Damping coefficient (non-negative)
     */
    inline void reset(int const NumDimensions, int const MaxNumPoints, EVectorXd const &LowerBounds, EVectorXd const &UpperBounds, double const DampingCoefficient = 1e-8);

  protected:
    /*!
     * \brief Set the Points object and computes the initial LU decomposition
     * 
     * \param Points          Initial points
     * \param FunctionValues  Values at the initial points
     * 
     * \throws std::runtime_error if the number of points are less than the dimension of the polynomial space
     */
    bool setPoints(EMatrixXd const &Points, EVectorXd const &FunctionValues);

  public:
    /*!
     * \brief Method for getting the current number of points
     * 
     * \returns int Current number of points
     */
    inline int numPoints() const override;

    /*!
     * \brief Method for getting the current number of points
     * 
     * \returns int Current number of points
     */
    inline int getNumPoints() const override;

    /*!
     * \brief Get the Number of Dimensions 
     * 
     * \returns int Number of dimensions
     */
    inline int dim() const override;

    /*!
     * \brief Get the Number of Dimensions 
     * 
     * \returns int Number of dimensions
     */
    inline int getNumDimensions() const override;

    /*!
     * \brief Method for getting the current points
     * 
     * \returns EMatrixXd Current points
     */
    inline EMatrixXd getCurrentPoints() const override;

    /*!
     * \brief Method for getting current point number i (0 is the first)
     * 
     * \param i Index number
     * 
     * \returns EVectorXd Point at index number i
     */
    inline EVectorXd getCurrentPoints(int i) const override;

    /*!
     * \brief Method for getting the values of the current points
     * 
     * \returns EVectorXd Values of current points 
     */
    inline EVectorXd getFunctionValues() const override;

    /*!
     * \brief Method for getting the value of current point number i (0 is the first)
     * 
     * \param i Index number
     * 
     * \returns double Value of point at index number i 
     */
    inline double getFunctionValues(int i) const override;

    /*!
     * \brief Get the Interpolation Coefficients object
     * 
     * Method for getting the radial basis function interpolation coefficients
     * 
     * \returns EVectorXd Interpolation coefficients.
     */
    inline EVectorXd getInterpolationCoefficients();

    /*!
     * \brief Method for adding one points with known value
     * 
     * \param Point          Point to be added
     * \param FunctionValue  Function value at the point
     * 
     * \throws std::runtime_error if capacity is exceeded
     */
    bool addPoint(EVectorXd const &Point, double const FunctionValue) override;

    /*!
     * \brief Method for adding multiple points with known values
     * 
     * \param Points          Points to be added 
     * \param FunctionValues  Function values at the points
     * 
     * \throws std::runtime_error if one point is supplied or if capacity is exceeded
     */
    bool addPoint(EMatrixXd const &Points, EVectorXd const &FunctionValues) override;

    /*!
     * \brief Method for evaluating the surrogate model at a point
     * 
     * \param Point  Point for which to evaluate the surrogate model
     * 
     * \returns double Value of the surrogate model at the point
     * 
     * \throws std::runtime_error if coefficients aren't updated
     */
    double evaluate(EVectorXd const &Point) const override;

    /*!
     * \brief Method for evaluating the surrogate model at a point with known distances
     * 
     * \param Point      Point for which to evaluate the surrogate model
     * \param Distances  Distances between the interpolation nodes and the point
     * 
     * \returns double Value of the surrogate model at the point
     * 
     * \throws std::runtime_error if coefficients aren't updated
     */
    double evaluate(EVectorXd const &Point, EVectorXd const &Distances) const override;

    /*!
     * \brief Method for evaluating the surrogate model at multiple points
     * 
     * \param Points Points for which to evaluate the surrogate model
     * 
     * \returns EVectorXd Values of the surrogate model at the points
     * 
     * \throws std::runtime_error if coefficients aren't updated
     */
    EVectorXd evaluate(EMatrixXd const &Points) const override;

    /*!
     * \brief Method for evaluating the surrogate model at multiple points with known distances
     * 
     * \param Points     Points for which to evaluate the surrogate model
     * \param Distances  Distances between the interpolation nodes and the points
     * 
     * \returns EVectorXd Value of the surrogate model at the point
     * 
     * \throws std::runtime_error if coefficients aren't updated
     */
    EVectorXd evaluate(EMatrixXd const &Points, EMatrixXd const &Distances) const override;

    /*!
     * \brief Method for evaluating the derivative of the surrogate model at a point
     * 
     * \param Point  Point for which to evaluate the derivative of the surrogate model
     * 
     * \returns EVectorXd Value of the derivative of the surrogate model at the point
     * 
     * \throws std::runtime_error if coefficients aren't updated
     */
    EVectorXd deriv(EVectorXd const &Point) const override;

    /*!
     * \brief Method for fitting the surrogate model
     * 
     */
    bool fit() override;

  protected:
    /*!
     * \brief Delete a radialBasisFunction object copy construction
     * 
     * Avoiding implicit generation of the copy constructor.
     */
    radialBasisFunction(radialBasisFunction<kernelType, polynomialTailType> const &) = delete;

    /*!
     * \brief Delete a radialBasisFunction object assignment
     * 
     * Avoiding implicit copy assignment.
     */
    radialBasisFunction<kernelType, polynomialTailType> &operator=(radialBasisFunction<kernelType, polynomialTailType> const &) = delete;

  protected:
    /*! Number of dimensions */
    int nDimensions;

    /*! Capacity */
    int maxNumPoints;

    /*! Current number of points */
    int nPoints;

    /*! Lower variable bounds */
    EVectorXd lowerBounds;

    /*! Upper variable bounds */
    EVectorXd upperBounds;

    /*! Damping added to the kernel to avoid ill-conditioning */
    double dampingCoefficient;

    /*! Dimensionality of the polynomial space */
    int polynomialSpaceDimension;

    /*! The radial kernel */
    kernelType kernel;

    /*! The polynomial tail */
    polynomialTailType tail;

    /*! 
     * The LU decomposition matrix: <br>
     * The upper-triangular part is U, the unit-lower-triangular part is L.
     */
    EMatrixXd matrixLU;

    /*! The vector representation of the permutation matrix P. */
    EVectorX<int> permutationP;

    /*! Coefficient vector */
    EVectorXd coefficientVector;

    /*! Function values */
    EVectorXd functionValues;

    /*! Interpolation nodes (centers) */
    EMatrixXd interpolationNodes;

    /*! True if the coefficients need to be recomputed */
    bool shouldRecomputeCoefficients;
};

template <class kernelType, class polynomialTailType>
radialBasisFunction<kernelType, polynomialTailType>::radialBasisFunction(int const NumDimensions,
                                                                         int const MaxNumPoints,
                                                                         double const DampingCoefficient) : radialBasisFunction(NumDimensions,
                                                                                                                                MaxNumPoints,
                                                                                                                                EVectorXd::Zero(NumDimensions),
                                                                                                                                EVectorXd::Ones(NumDimensions),
                                                                                                                                DampingCoefficient)
{
}

template <class kernelType, class polynomialTailType>
radialBasisFunction<kernelType, polynomialTailType>::radialBasisFunction(int const NumDimensions,
                                                                         int const MaxNumPoints,
                                                                         EVectorXd const &LowerBounds,
                                                                         EVectorXd const &UpperBounds,
                                                                         double const DampingCoefficient) : nDimensions(NumDimensions),
                                                                                                            maxNumPoints(MaxNumPoints),
                                                                                                            nPoints(0),
                                                                                                            lowerBounds(LowerBounds),
                                                                                                            upperBounds(UpperBounds),
                                                                                                            dampingCoefficient(DampingCoefficient)
{
#ifdef DEBUG
    gTimer.tic();
#endif
    polynomialSpaceDimension = tail.dimTail(nDimensions);
    if ((kernel.order() - 1) > tail.degree())
    {
        UMUQFAIL("Kernel and tail mismatch!");
    }
    matrixLU = EMatrixXd::Zero(maxNumPoints + polynomialSpaceDimension, maxNumPoints + polynomialSpaceDimension);
    permutationP = EVectorX<int>::Zero(maxNumPoints + polynomialSpaceDimension);
    coefficientVector = EVectorXd::Zero(maxNumPoints + polynomialSpaceDimension);
    functionValues = EVectorXd::Zero(maxNumPoints + polynomialSpaceDimension);
    interpolationNodes = EMatrixXd::Zero(nDimensions, maxNumPoints);
    shouldRecomputeCoefficients = false;
#ifdef DEBUG
    gTimer.toc("radialBasisFunction->construction");
#endif
}

template <class kernelType, class polynomialTailType>
radialBasisFunction<kernelType, polynomialTailType>::radialBasisFunction(radialBasisFunction<kernelType, polynomialTailType> &&other)
{
    nDimensions = other.nDimensions;
    maxNumPoints = other.maxNumPoints;
    nPoints = other.nPoints;
    lowerBounds = std::move(other.lowerBounds);
    upperBounds = std::move(other.upperBounds);
    dampingCoefficient = other.dampingCoefficient;
    polynomialSpaceDimension = other.polynomialSpaceDimension;
    matrixLU = std::move(other.matrixLU);
    permutationP = std::move(other.permutationP);
    coefficientVector = std::move(other.coefficientVector);
    functionValues = std::move(other.functionValues);
    interpolationNodes = std::move(other.interpolationNodes);
    shouldRecomputeCoefficients = other.shouldRecomputeCoefficients;
}

template <class kernelType, class polynomialTailType>
radialBasisFunction<kernelType, polynomialTailType> &radialBasisFunction<kernelType, polynomialTailType>::operator=(radialBasisFunction<kernelType, polynomialTailType> &&other)
{
    nDimensions = other.nDimensions;
    maxNumPoints = other.maxNumPoints;
    nPoints = other.nPoints;
    lowerBounds = std::move(other.lowerBounds);
    upperBounds = std::move(other.upperBounds);
    dampingCoefficient = other.dampingCoefficient;
    polynomialSpaceDimension = other.polynomialSpaceDimension;
    matrixLU = std::move(other.matrixLU);
    permutationP = std::move(other.permutationP);
    coefficientVector = std::move(other.coefficientVector);
    functionValues = std::move(other.functionValues);
    interpolationNodes = std::move(other.interpolationNodes);
    shouldRecomputeCoefficients = other.shouldRecomputeCoefficients;
    return *this;
}

template <class kernelType, class polynomialTailType>
radialBasisFunction<kernelType, polynomialTailType>::~radialBasisFunction() {}

template <class kernelType, class polynomialTailType>
inline void radialBasisFunction<kernelType, polynomialTailType>::reset() { nPoints = 0; }

template <class kernelType, class polynomialTailType>
bool radialBasisFunction<kernelType, polynomialTailType>::setPoints(EMatrixXd const &Points, EVectorXd const &FunctionValues)
{
#ifdef DEBUG
    gTimer.tic();
#endif

    // Map point to be in the unit box
    EMatrixXd points = scaleToUnitBox(Points, lowerBounds, upperBounds);

    nPoints = static_cast<int>(points.cols());

    if (nPoints < polynomialSpaceDimension)
    {
        UMUQFAILRETURN("Current number of points = ", nPoints, " < ", polynomialSpaceDimension, " (polynomial dimension), Not enough points!");
    }

    functionValues.segment(polynomialSpaceDimension, nPoints) = FunctionValues;

    EMatrixXd px = tail.evaluate(points);

    EMatrixXd phi = kernel.evaluate(L2Distance(points));

    auto const n = nPoints + polynomialSpaceDimension;

    EMatrixXd A = EMatrixXd::Zero(n, n);

    A.block(polynomialSpaceDimension, 0, nPoints, polynomialSpaceDimension) = px.transpose();
    A.block(0, polynomialSpaceDimension, polynomialSpaceDimension, nPoints) = px;
    A.block(polynomialSpaceDimension, polynomialSpaceDimension, nPoints, nPoints) = phi;

    // REGULARIZATION
    A += dampingCoefficient * EMatrixXd::Identity(n, n);

    // Compute the initial LU factorization of A
    Eigen::PartialPivLU<EMatrixXd> lu(A);

    matrixLU.block(0, 0, n, n) = lu.matrixLU();

    permutationP.segment(0, n) = lu.permutationP().indices();

    interpolationNodes.block(0, 0, points.rows(), points.cols()) = points;

    shouldRecomputeCoefficients = true;

#ifdef DEBUG
    gTimer.toc("radialBasisFunction->setPoints");
#endif
    return true;
}

template <class kernelType, class polynomialTailType>
inline int radialBasisFunction<kernelType, polynomialTailType>::numPoints() const { return nPoints; }

template <class kernelType, class polynomialTailType>
inline int radialBasisFunction<kernelType, polynomialTailType>::getNumPoints() const { return nPoints; }

template <class kernelType, class polynomialTailType>
inline int radialBasisFunction<kernelType, polynomialTailType>::dim() const { return nDimensions; }

template <class kernelType, class polynomialTailType>
inline int radialBasisFunction<kernelType, polynomialTailType>::getNumDimensions() const { return nDimensions; }

template <class kernelType, class polynomialTailType>
inline EMatrixXd radialBasisFunction<kernelType, polynomialTailType>::getCurrentPoints() const
{
    return scaleToHyperCube<double, true>(interpolationNodes.block(0, 0, nDimensions, nPoints), lowerBounds, upperBounds);
}

template <class kernelType, class polynomialTailType>
inline EVectorXd radialBasisFunction<kernelType, polynomialTailType>::getCurrentPoints(int i) const
{
    return scaleToHyperCube<double>(static_cast<EVectorXd>(interpolationNodes.col(i)), lowerBounds, upperBounds);
}

template <class kernelType, class polynomialTailType>
inline EVectorXd radialBasisFunction<kernelType, polynomialTailType>::getFunctionValues() const
{
    return functionValues.segment(polynomialSpaceDimension, nPoints);
}

template <class kernelType, class polynomialTailType>
inline double radialBasisFunction<kernelType, polynomialTailType>::getFunctionValues(int i) const
{
    return functionValues(polynomialSpaceDimension + i);
}

template <class kernelType, class polynomialTailType>
inline EVectorXd radialBasisFunction<kernelType, polynomialTailType>::getInterpolationCoefficients()
{
    if (shouldRecomputeCoefficients)
    {
        UMUQFAIL("RBF not updated!");
    }
    return coefficientVector;
}

template <class kernelType, class polynomialTailType>
bool radialBasisFunction<kernelType, polynomialTailType>::addPoint(EVectorXd const &Point, double const FunctionValue)
{
    if (!nPoints)
    {
        EVectorXd fVal(1);
        fVal << FunctionValue;
        return setPoints(static_cast<EMatrixXd>(Point), fVal);
    }
    if (nPoints + 1 > maxNumPoints)
    {
        UMUQFAILRETURN("Capacity exceeded");
    }
#ifdef DEBUG
    gTimer.tic();
#endif

    // Map point to be in the unit box
    EVectorXd point = scaleToUnitBox(Point, lowerBounds, upperBounds);

    int const n = polynomialSpaceDimension + nPoints;

    EVectorXd vx(n);
    vx << tail.evaluate(point), kernel.evaluate(L2Distance<double>(point, interpolationNodes.block(0, 0, nDimensions, nPoints)));

    // Step 1
    EVectorXd u12 = Eigen::PermutationWrapper<EVectorX<int>>(permutationP.head(n)) * vx;
    EVectorXd l21 = vx;

    // Step 2
    matrixLU.block(0, 0, n, n).template triangularView<Eigen::UnitLower>().solveInPlace(u12);

    // Step 3
    matrixLU.block(0, 0, n, n).template triangularView<Eigen::Upper>().solveInPlace(l21);

    double u22 = kernel.phiZero() + dampingCoefficient - u12.dot(l21);

    matrixLU.block(n, 0, 1, n) = l21.transpose();
    matrixLU.block(0, n, n, 1) = u12;
    matrixLU(n, n) = u22;
    permutationP(n) = n;

    // Update F and add the centers
    functionValues(n) = FunctionValue;

    interpolationNodes.col(nPoints) = point;

    nPoints++;

    shouldRecomputeCoefficients = true;

#ifdef DEBUG
    gTimer.toc("radialBasisFunction->addPoint");
#endif
    return true;
}

template <class kernelType, class polynomialTailType>
bool radialBasisFunction<kernelType, polynomialTailType>::addPoint(EMatrixXd const &Points, EVectorXd const &FunctionValues)
{
    // check for the correct number of points and function values at those points
    UMUQ_assert(Points.cols() == FunctionValues.rows());

    if (!nPoints)
    {
        return setPoints(Points, FunctionValues);
    }

    int const newNumPoints = static_cast<int>(FunctionValues.rows());
    if (newNumPoints < 2)
    {
        if (newNumPoints)
        {
            return addPoint(Points.col(0), FunctionValues(0));
        }
        UMUQFAILRETURN("There is no input data!");
    }

    if (nPoints + newNumPoints > maxNumPoints)
    {
        UMUQFAILRETURN("Capacity exceeded ! \n Number of current points ", nPoints, " + number of new points ", newNumPoints, " > the available capacity ", maxNumPoints);
    }

#ifdef DEBUG
    gTimer.tic();
#endif

    // Map point to be in the unit box
    EMatrixXd points = scaleToUnitBox(Points, lowerBounds, upperBounds);

    int const n = polynomialSpaceDimension + nPoints;

    EMatrixXd px = tail.evaluate(points);
    EMatrixXd B = EMatrixXd::Zero(n, newNumPoints);

    B.block(0, 0, polynomialSpaceDimension, newNumPoints) = px;
    B.block(polynomialSpaceDimension, 0, nPoints, newNumPoints) = kernel.evaluate(L2Distance<double, true>(interpolationNodes.block(0, 0, nDimensions, nPoints), points));

    EMatrixXd K = kernel.evaluate(L2Distance(points));

    // REGULARIZATION
    K += dampingCoefficient * EMatrixXd::Identity(newNumPoints, newNumPoints);

    // Update the LU factorization
    // Step 1
    EMatrixXd U12 = Eigen::PermutationWrapper<EVectorX<int>>(permutationP.head(n)) * B;
    EMatrixXd L21 = U12;

    // Step 2
    matrixLU.block(0, 0, n, n).template triangularView<Eigen::UnitLower>().solveInPlace(U12);

    // Step 3
    matrixLU.block(0, 0, n, n).template triangularView<Eigen::Upper>().solveInPlace(L21);

    EMatrixXd C;

    // Standard Cholesky decomposition (LL^T)
    Eigen::LLT<EMatrixXd> llt(K - L21.transpose() * U12);

    if (llt.info() == Eigen::Success)
    {
        // retrieve factor L  in the decomposition
        C = llt.matrixL();
    }
    else
    {
        UMUQWARNING("Cholesky factorization failed, computing new LU from scratch...");

        // Add new points
        functionValues.segment(n, newNumPoints) = FunctionValues;
        interpolationNodes.block(0, nPoints, nDimensions, newNumPoints) = points;
        nPoints += newNumPoints;
        // Build LU from scratch
        return setPoints(getCurrentPoints(), functionValues.segment(polynomialSpaceDimension, nPoints));
    }

    matrixLU.block(n, 0, newNumPoints, n).template triangularView<Eigen::UnitLower>() = L21;
    matrixLU.block(n, n, newNumPoints, newNumPoints).template triangularView<Eigen::UnitLower>() = C;
    matrixLU.block(0, n, n, newNumPoints).template triangularView<Eigen::Upper>() = U12;
    matrixLU.block(n, n, newNumPoints, newNumPoints).template triangularView<Eigen::Upper>() = C.transpose();

    permutationP.segment(n, newNumPoints) = EVectorX<int>::LinSpaced(newNumPoints, n, n + newNumPoints - 1);

    // Update F and add the centers
    functionValues.segment(n, newNumPoints) = FunctionValues;

    interpolationNodes.block(0, nPoints, nDimensions, newNumPoints) = points;

    nPoints += newNumPoints;

    shouldRecomputeCoefficients = true;

#ifdef DEBUG
    gTimer.toc("radialBasisFunction->addPoint");
#endif

    return true;
}

template <class kernelType, class polynomialTailType>
double radialBasisFunction<kernelType, polynomialTailType>::evaluate(EVectorXd const &Point) const
{
    if (shouldRecomputeCoefficients)
    {
        UMUQFAIL("radialBasisFunction is not updated. You need to call fit() first");
    }
    // Map point to be in the unit box
    EVectorXd point = scaleToUnitBox(Point, lowerBounds, upperBounds);
    EVectorXd px = tail.evaluate(point);
    EVectorXd phi = kernel.evaluate(L2Distance<double>(point, interpolationNodes.block(0, 0, nDimensions, nPoints)));
    return coefficientVector.head(polynomialSpaceDimension).dot(px) + coefficientVector.segment(polynomialSpaceDimension, nPoints).dot(phi);
}

template <class kernelType, class polynomialTailType>
double radialBasisFunction<kernelType, polynomialTailType>::evaluate(EVectorXd const &Point, EVectorXd const &Distances) const
{
    if (shouldRecomputeCoefficients)
    {
        UMUQFAIL("radialBasisFunction is not updated. You need to call fit() first");
    }
    if ((lowerBounds.array() != 0).any() || (upperBounds.array() != 1).any())
    {
        UMUQWARNING("radialBasisFunction uses internal scaling so distances have to be recomputed");
        return evaluate(Point);
    }
    // check for the correct number of points and distances vector
    UMUQ_assert(nPoints == Distances.rows());
    // Map point to be in the unit box
    EVectorXd point = scaleToUnitBox(Point, lowerBounds, upperBounds);
    EVectorXd px = tail.evaluate(point);
    EVectorXd phi = kernel.evaluate(Distances);
    return coefficientVector.head(polynomialSpaceDimension).dot(px) + coefficientVector.segment(polynomialSpaceDimension, nPoints).dot(phi);
}

template <class kernelType, class polynomialTailType>
EVectorXd radialBasisFunction<kernelType, polynomialTailType>::evaluate(EMatrixXd const &Points) const
{
    // Check for the correct dimensionality
    UMUQ_assert(Points.rows() == nDimensions);
    if (shouldRecomputeCoefficients)
    {
        UMUQFAIL("radialBasisFunction is not updated. You need to call fit() first");
    }
    // Map point to be in the unit box
    EMatrixXd points = scaleToUnitBox(Points, lowerBounds, upperBounds);
    EMatrixXd px = tail.evaluate(points);
    EMatrixXd phi = kernel.evaluate(L2Distance<double, true>(points, interpolationNodes.block(0, 0, nDimensions, nPoints)));
    return px.transpose() * coefficientVector.head(polynomialSpaceDimension) + phi * coefficientVector.segment(polynomialSpaceDimension, nPoints);
}

template <class kernelType, class polynomialTailType>
EVectorXd radialBasisFunction<kernelType, polynomialTailType>::evaluate(EMatrixXd const &Points, EMatrixXd const &Distances) const
{
    // Check for the correct dimensions
    UMUQ_assert(Points.cols() == Distances.rows());
    if (shouldRecomputeCoefficients)
    {
        UMUQFAIL("radialBasisFunction is not updated. You need to call fit() first");
    }
    if ((lowerBounds.array() != 0).any() || (upperBounds.array() != 1).any())
    {
        UMUQWARNING("radialBasisFunction uses internal scaling so distances have to be recomputed");
        return evaluate(Points);
    }
    // Map point to be in the unit box
    EMatrixXd points = scaleToUnitBox(Points, lowerBounds, upperBounds);
    EMatrixXd px = tail.evaluate(points);
    EMatrixXd phi = kernel.evaluate(Distances);
    return px.transpose() * coefficientVector.head(polynomialSpaceDimension) + phi * coefficientVector.segment(polynomialSpaceDimension, nPoints);
}

template <class kernelType, class polynomialTailType>
EVectorXd radialBasisFunction<kernelType, polynomialTailType>::deriv(EVectorXd const &Point) const
{
    if (shouldRecomputeCoefficients)
    {
        UMUQFAIL("radialBasisFunction is not updated. You need to call fit() first");
    }
    // Map point to be in the unit box
    EVectorXd point = scaleToUnitBox(Point, lowerBounds, upperBounds);

    EMatrixXd dpx = tail.deriv(point);
    EVectorXd Distances = L2Distance<double>(point, interpolationNodes.block(0, 0, nDimensions, nPoints));
    // Better safe than sorry
    for (int i = 0; i < nPoints; ++i)
    {
        if (Distances(i) < 1e-10)
        {
            Distances(i) = 1e-10;
        }
    }
    EMatrixXd dsx = -interpolationNodes.block(0, 0, nDimensions, nPoints);
    dsx.colwise() += point;
    dsx.array().rowwise() *= coefficientVector.segment(polynomialSpaceDimension, nPoints).cwiseProduct(kernel.deriv(Distances)).cwiseProduct(Distances.cwiseInverse()).array().transpose();
    return dsx.rowwise().sum() + dpx.transpose() * coefficientVector.head(polynomialSpaceDimension);
}

template <class kernelType, class polynomialTailType>
bool radialBasisFunction<kernelType, polynomialTailType>::fit()
{
    if (nPoints < polynomialSpaceDimension)
    {
        UMUQFAILRETURN("Current number of points = ", nPoints, " < ", polynomialSpaceDimension, " (polynomial dimension), Not enough points!");
    }
#ifdef DEBUG
    gTimer.tic();
#endif
    if (shouldRecomputeCoefficients)
    {
        shouldRecomputeCoefficients = false;

        int const n = nPoints + polynomialSpaceDimension;

        /* The decomposition PA = LU.
         * So we proceed as follows:
         * Step 1: compute c = Pb.
         * Step 2: replace c by the solution x to Lx = c.
         * Step 3: replace c by the solution x to Ux = c.
         */

        // Step 1
        coefficientVector.head(n) = Eigen::PermutationWrapper<EVectorX<int>>(permutationP.head(n)) * functionValues.head(n);

        // Step 2
        matrixLU.block(0, 0, n, n).template triangularView<Eigen::UnitLower>().solveInPlace(coefficientVector.head(n));

        // Step 3
        matrixLU.block(0, 0, n, n).template triangularView<Eigen::Upper>().solveInPlace(coefficientVector.head(n));
    }
#ifdef DEBUG
    gTimer.toc("radialBasisFunction->fit");
#endif
    return true;
}

/*! \class radialBasisFunctionCap
 * \ingroup Surrogate_Module
 *
 * \brief Capped radial basis function interpolant
 * 
 * \tparam kernelType         Radial basis function kernel type (Cubic is default)
 * \tparam polynomialTailType Polynomial tail type (Linear is default)
 * 
 * \author David Eriksson, dme65@cornell.edu
 * 
 * This is a capped version of the RBF interpolant that is useful in cases
 * where there are large function values. This version replaces all of the
 * function values that are above the median of the function values by
 * the value of the median.
 */
template <class kernelType, class polynomialTailType>
class radialBasisFunctionCap : public radialBasisFunction<kernelType, polynomialTailType>
{
  public:
    /*!
     * \brief Construct a new radialBasisFunctionCap object
     * 
     * \param MaxNumPoints        Capacity, maximum number of points
     * \param NumDimensions       Number of dimensions
     * \param DampingCoefficient  Damping coefficient (non-negative)
     */
    radialBasisFunctionCap(int const NumDimensions,
                           int const MaxNumPoints,
                           double const DampingCoefficient = 1e-8);

    /*!
     * \brief Construct a new radialBasisFunctionCap object
     * 
     * \param MaxNumPoints        Capacity, maximum number of points
     * \param NumDimensions       Number of dimensions
     * \param LowerBounds         Lower variable bounds
     * \param UpperBounds         Upper variable bounds
     * \param DampingCoefficient  Damping coefficient (non-negative)
     */
    radialBasisFunctionCap(int const NumDimensions,
                           int const MaxNumPoints,
                           EVectorXd const &LowerBounds,
                           EVectorXd const &UpperBounds,
                           double const DampingCoefficient = 1e-8);

    /*!
     * \brief Destroy the radialBasisFunctionCap object
     * 
     */
    ~radialBasisFunctionCap();

    /*!
     * \brief Method for fitting the surrogate model
     * 
     */
    bool fit();
};

template <class kernelType, class polynomialTailType>
radialBasisFunctionCap<kernelType, polynomialTailType>::radialBasisFunctionCap(int const NumDimensions,
                                                                               int const MaxNumPoints,
                                                                               double const DampingCoefficient) : radialBasisFunction<kernelType, polynomialTailType>(NumDimensions,
                                                                                                                                                                      MaxNumPoints,
                                                                                                                                                                      DampingCoefficient) {}

template <class kernelType, class polynomialTailType>
radialBasisFunctionCap<kernelType, polynomialTailType>::radialBasisFunctionCap(int const NumDimensions,
                                                                               int const MaxNumPoints,
                                                                               EVectorXd const &LowerBounds,
                                                                               EVectorXd const &UpperBounds,
                                                                               double const DampingCoefficient) : radialBasisFunction<kernelType, polynomialTailType>(NumDimensions,
                                                                                                                                                                      MaxNumPoints,
                                                                                                                                                                      LowerBounds,
                                                                                                                                                                      UpperBounds,
                                                                                                                                                                      DampingCoefficient) {}

template <class kernelType, class polynomialTailType>
radialBasisFunctionCap<kernelType, polynomialTailType>::~radialBasisFunctionCap() {}

template <class kernelType, class polynomialTailType>
bool radialBasisFunctionCap<kernelType, polynomialTailType>::fit()
{
    if (this->nPoints < this->polynomialSpaceDimension)
    {
        UMUQFAILRETURN("Current number of points = ", this->nPoints, " < ", this->polynomialSpaceDimension, " (polynomial dimension), Not enough points!");
    }
#ifdef DEBUG
    gTimer.tic();
#endif
    if (this->shouldRecomputeCoefficients)
    {
        int const n = this->nPoints + this->polynomialSpaceDimension;

        double functionValuesMedian;

        // Computes the median
        {
            // We do partial sorting algorithm that rearranges elements
            std::vector<double> functionValuesVector(n);
            // Copy the function values to the std vector
            EMap<EVectorXd>(functionValuesVector.data(), this->functionValues.head(n));
            std::nth_element(functionValuesVector.begin(), functionValuesVector.begin() + n / 2, functionValuesVector.end());
            functionValuesMedian = functionValuesVector[n / 2];
        }

        for (int i = this->polynomialSpaceDimension; i < n; ++i)
        {
            // Apply the capping
            if (this->functionValues(i) > functionValuesMedian)
            {
                this->functionValues(i) = functionValuesMedian;
            }
        }

        /* The decomposition PA = LU.
         * So we proceed as follows:
         * Step 1: compute c = Pb.
         * Step 2: replace c by the solution x to Lx = c.
         * Step 3: replace c by the solution x to Ux = c.
         */

        // Step 1
        this->coefficientVector.head(n) = Eigen::PermutationWrapper<EVectorX<int>>(this->permutationP.head(n)) * this->functionValues.head(n);

        // Step 2
        this->matrixLU.block(0, 0, n, n).template triangularView<Eigen::UnitLower>().solveInPlace(this->coefficientVector.head(n));

        // Step 3
        this->matrixLU.block(0, 0, n, n).template triangularView<Eigen::Upper>().solveInPlace(this->coefficientVector.head(n));

        this->shouldRecomputeCoefficients = false;
    }
#ifdef DEBUG
    gTimer.toc("radialBasisFunctionCap->fit");
#endif
    return true;
}

} // namespace umuq

#endif // UMUQ_RADIALBASISFUNCTION

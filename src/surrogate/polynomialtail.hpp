#ifndef UMUQ_POLYNOMIALTAIL_H
#define UMUQ_POLYNOMIALTAIL_H

#include "numerics/eigenlib.hpp"

namespace umuq
{

/*! \file polynomialtail.hpp
 * \ingroup 
 * 
 * \brief Implementation of the polynomial tail.
 *
 * This file contains minor addition to the original rbf.h
 * source code made available under the following license:
 *
 * \author David Eriksson, dme65@cornell.edu
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

/*! \enum polynomialTailTypes
 * \ingroup Surrogate_Module
 * 
 * \brief Radial basis function kernel types currently supported in %UMUQ
 */
enum class polynomialTailTypes
{
    /*! \link  LINEAR \endlink */
    LINEAR,
    /*! \link  CONSTANT \endlink */
    CONSTANT
};

/*! \class polynomialTail
 * \ingroup Surrogate_Module
 *
 * \brief Abstract class for a polynomial tail
 * 
 * This is the abstract class that should be used as a Base class for all Polynomial tails 
 * 
 * \author David Eriksson, dme65@cornell.edu
 */
class polynomialTail
{
  public:
    /*!
     * \brief Construct a new polynomial Tail object
     * 
     */
    polynomialTail();

    /*!
     * \brief Destroy the polynomial Tail object
     * 
     */
    ~polynomialTail();

    /*!
     * \brief Method for getting the degree of the tail
     * 
     * \returns int Degree of the tail
     */
    virtual inline int degree() const = 0;

    /*!
     * \brief Method for the dimensionality of the polynomial space
     * 
     * \param dim Dimensionality of the input
     * 
     * \returns int Dimensionality of the polynomial space (number of basis functions)
     */
    virtual inline int dimTail(int dim) const = 0;

    /*!
     * \brief Method for evaluating the monomial basis function for a given point
     * 
     * \param point Point for which to evaluate the monomial basis function of the tail
     * 
     * \returns EVectorXd Value of the monomial basis functions at the point
     */
    virtual inline EVectorXd eval(const EVectorXd &point) const = 0;

    /*!
     * \brief Method for evaluating the monomial basis function for multiple points
     * 
     * \param points Points for which to evaluate the monomial basis function of the tail
     * 
     * \returns EMatrixXd Values of the monomial basis functions at the points
     */
    virtual inline EMatrixXd eval(const EMatrixXd &points) const = 0;

    /*!
     * \brief Method for evaluating the derivative of the monomial basis function for a given point
     * 
     * \param point Point for which to evaluate the derivative of the monomial basis function of the tail
     * 
     * \returns EMatrixXd Values of the derivative of the monomial basis functions at the point
     */
    virtual inline EMatrixXd deriv(const EVectorXd &point) const = 0;
};

polynomialTail::polynomialTail() {}

polynomialTail::~polynomialTail() {}

/*! \class linearPolynomialTail
 * \ingroup Surrogate_Module
 *
 * \brief Linear polynomial tail
 * 
 * This is an implementation of the linear polynomial tail with basis 
 * \f$\{1,x_1,x_2,\dots,x_d\}\f$ of degree 1. Popular to use with the
 * Cubic or the TPS kernel.
 *
 * \author David Eriksson, dme65@cornell.edu
 */
class linearPolynomialTail : public polynomialTail
{
  public:
    /*!
     * \brief Construct a new linear Polynomial Tail object
     * 
     */
    linearPolynomialTail();

    /*!
     * \brief Destroy the linear Polynomial Tail object
     * 
     */
    ~linearPolynomialTail();

    /*!
     * \brief Method for getting the degree of the tail
     * 
     * \returns int Degree of the tail
     */
    inline int degree() const;

    /*!
     * \brief Method for the dimensionality of the polynomial space
     * 
     * \param dim Dimensionality of the input
     * 
     * \returns int Dimensionality of the polynomial space (number of basis functions)
     */
    inline int dimTail(int dim) const;

    /*!
     * \brief Method for evaluating the monomial basis function for a given point
     * 
     * \param point Point for which to evaluate the monomial basis function of the tail
     * 
     * \returns EVectorXd Value of the monomial basis functions at the point
     */
    inline EVectorXd eval(const EVectorXd &x) const;

    /*!
     * \brief Method for evaluating the monomial basis function for multiple points
     * 
     * \param points Points for which to evaluate the monomial basis function of the tail
     * 
     * \returns EMatrixXd Values of the monomial basis functions at the points
     */
    inline EMatrixXd eval(const EMatrixXd &X) const;

    /*!
     * \brief Method for evaluating the derivative of the monomial basis function for a given point
     * 
     * \param point Point for which to evaluate the derivative of the monomial basis function of the tail
     * 
     * \returns EMatrixXd Values of the derivative of the monomial basis functions at the point
     */
    inline EMatrixXd deriv(const EVectorXd &x) const;

  private:
    /*! Degree of the polynomial tail */
    int mDegree;
};

linearPolynomialTail::linearPolynomialTail() : mDegree(1) {}

linearPolynomialTail::~linearPolynomialTail() {}

inline int linearPolynomialTail::degree() const { return mDegree; }

inline int linearPolynomialTail::dimTail(int dim) const { return 1 + dim; }

inline EVectorXd linearPolynomialTail::eval(const EVectorXd &x) const
{
    EVectorXd tail(x.rows() + 1);
    tail << double{1}, x;
    return tail;
}

inline EMatrixXd linearPolynomialTail::eval(const EMatrixXd &X) const
{
    EMatrixXd Y(X.rows() + 1, X.cols());
    Y << ERowVectorXd::Ones(X.cols()), X;
    return Y;
}

inline EMatrixXd linearPolynomialTail::deriv(const EVectorXd &x) const
{
    auto const nSize = x.rows();
    EMatrixXd Y(nSize + 1, nSize);
    Y << ERowVectorXd::Zero(nSize), EMatrixXd::Identity(nSize, nSize);
    return Y;
}

/*! \class constantPolynomialTail
 * \ingroup Surrogate_Module
 * 
 * \brief Constant polynomial tail
 * 
 * This is an implementation of the constant polynomial tail with basis 
 * \f$\{1\}\f$ of degree 0. Popular to use with the linear kernel.
 *
 * \author David Eriksson, dme65@cornell.edu
 */
class constantPolynomialTail : public polynomialTail
{
  public:
    /*!
     * \brief Construct a new constant Polynomial Tail object
     * 
     */
    constantPolynomialTail();

    /*!
     * \brief Destroy the constant Polynomial Tail object
     * 
     */
    ~constantPolynomialTail();

    /*!
     * \brief Method for getting the degree of the tail
     * 
     * \returns int Degree of the tail
     */
    inline int degree() const;

    /*!
     * \brief Method for the dimensionality of the polynomial space
     * 
     * \param dim Dimensionality of the input
     * 
     * \returns int Dimensionality of the polynomial space (number of basis functions)
     */
    inline int dimTail(int dim) const;

    /*!
     * \brief Method for evaluating the monomial basis function for a given point
     * 
     * \param point Point for which to evaluate the monomial basis function of the tail
     * 
     * \returns EVectorXd Value of the monomial basis functions at the point
     */
    inline EVectorXd eval(const EVectorXd &x) const;

    /*!
     * \brief Method for evaluating the monomial basis function for multiple points
     * 
     * \param points Points for which to evaluate the monomial basis function of the tail
     * 
     * \returns EMatrixXd Values of the monomial basis functions at the points
     */
    inline EMatrixXd eval(const EMatrixXd &X) const;

    /*!
     * \brief Method for evaluating the derivative of the monomial basis function for a given point
     * 
     * \param point Point for which to evaluate the derivative of the monomial basis function of the tail
     * 
     * \returns EMatrixXd Values of the derivative of the monomial basis functions at the point
     */
    inline EMatrixXd deriv(const EVectorXd &x) const;

  private:
    /*! Degree of the polynomial tail */
    int mDegree;
};

constantPolynomialTail::constantPolynomialTail() : mDegree(0) {}

constantPolynomialTail::~constantPolynomialTail() {}

inline int constantPolynomialTail::degree() const { return mDegree; }

inline int constantPolynomialTail::dimTail(int dim) const { return 1; }

inline EVectorXd constantPolynomialTail::eval(const EVectorXd &x) const { return EMatrixXd::Ones(1, 1); }

inline EMatrixXd constantPolynomialTail::eval(const EMatrixXd &X) const { return EMatrixXd::Ones(1, X.cols()); }

inline EMatrixXd constantPolynomialTail::deriv(const EVectorXd &x) const { return EMatrixXd::Zero(1, x.rows()); }

} // namespace umuq

#endif // UMUQ_POLYNOMIALTAIL

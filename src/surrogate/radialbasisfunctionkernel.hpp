#ifndef UMUQ_RADIALBASISFUNCTIONKERNEL_H
#define UMUQ_RADIALBASISFUNCTIONKERNEL_H

#include "numerics/eigenlib.hpp"

namespace umuq
{

/*! \file radialbasisfunctionkernel.hpp
 * \ingroup 
 * 
 * \brief Implementation of the Radial basis function kernel.
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

/*! \enum radialBasisFunctionKernelTypes
 * \ingroup Surrogate_Module
 * 
 * \brief Radial basis function kernel types currently supported in %UMUQ
 */
enum class radialBasisFunctionKernelTypes
{
    /*! \link umuq::linearKernel LINEAR \endlink */
    LINEAR,
    /*! \link umuq::cubicKernel CUBIC \endlink */
    CUBIC,
    /*! \link umuq::thinPlateKernel THINPLATE \endlink */
    THINPLATE
};

/*! \class radialBasisFunctionKernel
 * \ingroup Surrogate_Module
 * 
 * \brief Abstract class for a radial basis function kernel
 * 
 * This is the abstract class that should be used as a Base class for all RBF kernels
 *
 * \author David Eriksson, dme65@cornell.edu
 */
class radialBasisFunctionKernel
{
  public:
    /*!
     * \brief Construct a new radialBasisFunctionKernel object
     * 
     */
    radialBasisFunctionKernel();

    /*!
     * \brief Destroy the radialBasisFunctionKernel object
     * 
     */
    ~radialBasisFunctionKernel();

    /*!
     * \brief Method for getting the order of the kernel
     * 
     * \returns int  Order of the kernel 
     */
    virtual inline int order() const = 0;

    /*!
     * \brief Method for getting the value of the kernel at 0
     * 
     * \returns int Value of kernel at 0
     */
    virtual inline int phiZero() const = 0;

    /*!
     * \brief Method for evaluating the kernel for a given Distance
     * 
     * \param Distance  Distance for which to evaluate the kernel
     * 
     * \returns double Value of kernel at Distance
     */
    virtual inline double eval(double Distance) const = 0;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a given Distance 
     * 
     * \param Distance Distance for which to evaluate the derivative of the kernel
     * 
     * \returns double Derivative of kernel at Distance
     */
    virtual inline double deriv(double Distance) const = 0;

    /*!
     * \brief Method for evaluating the kernel for a matrix of distances
     * 
     * \param dists  Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EMatrixXd Values of kernel at dists
     */
    virtual inline EMatrixXd eval(const EMatrixXd &dists) const = 0;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a matrix of distances
     * 
     * \param dists Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EMatrixXd Values of the derivative of the kernel at dists
     */
    virtual inline EMatrixXd deriv(const EMatrixXd &dists) const = 0;
};

radialBasisFunctionKernel::radialBasisFunctionKernel() {}

radialBasisFunctionKernel::~radialBasisFunctionKernel() {}

/*! \class linearKernel
 * \ingroup Surrogate_Module
 *
 * \brief Linear kernel
 * 
 * This is an implementation of the linear kernel \f$\varphi(r)=r\,\log(r)\f$ which is of order 1.
 *
 * \author David Eriksson, dme65@cornell.edu
 */
class linearKernel : public radialBasisFunctionKernel
{
  public:
    /*!
     * \brief Construct a new linear Kernel object
     * 
     */
    linearKernel();

    /*!
     * \brief Destroy the linear Kernel object
     * 
     */
    ~linearKernel();

    /*!
     * \brief Method for getting the order of the kernel
     * 
     * \returns int  Order of the kernel 
     */
    inline int order() const;

    /*!
     * \brief Method for getting the value of the kernel at 0
     * 
     * \returns int Value of kernel at 0
     */
    inline int phiZero() const;

    /*!
     * \brief Method for evaluating the kernel for a given Distance
     * 
     * \param Distance  Distance for which to evaluate the kernel
     * 
     * \returns double Value of kernel at Distance
     */
    inline double eval(double Distance) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a given Distance 
     * 
     * \param Distance Distance for which to evaluate the derivative of the kernel
     * 
     * \returns double Derivative of kernel at Distance
     */

    inline double deriv(double Distance) const;

    /*!
     * \brief Method for evaluating the kernel for a matrix of distances
     * 
     * \param dists  Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EMatrixXd Values of kernel at dists
     */
    inline EMatrixXd eval(const EMatrixXd &dists) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a matrix of distances
     * 
     * \param dists Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EMatrixXd Values of the derivative of the kernel at dists
     */
    inline EMatrixXd deriv(const EMatrixXd &dists) const;

  private:
    /*! Value of the cubic kernel at 0 */
    int mPhiZero;

    /*! Order of the cubic kernel */
    int mOrder;
};

linearKernel::linearKernel() : mPhiZero(0), mOrder(1) {}

linearKernel::~linearKernel() {}

inline int linearKernel::order() const
{
    return mOrder;
}

inline int linearKernel::phiZero() const
{
    return mPhiZero;
}

inline double linearKernel::eval(double Distance) const
{
    return Distance;
}

inline double linearKernel::deriv(double Distance) const
{
    return 1.0;
}

inline EMatrixXd linearKernel::eval(const EMatrixXd &dists) const
{
    return dists;
}

inline EMatrixXd linearKernel::deriv(const EMatrixXd &dists) const
{
    return EMatrixXd::Ones(dists.rows(), dists.cols());
}

/*! \class cubicKernel
 * \ingroup Surrogate_Module
 *
 * \brief Cubic kernel for a radial basis function
 *
 * This is an implementation of the popular cubic kernel \f$\varphi(r)=r^3\f$ which is of order 2.
 * 
 * \author David Eriksson, dme65@cornell.edu 
 */
class cubicKernel : public radialBasisFunctionKernel
{
  public:
    /*!
     * \brief Construct a new cubic Kernel object
     * 
     */
    cubicKernel();

    /*!
     * \brief Destroy the cubic Kernel object
     * 
     */
    ~cubicKernel();

    /*!
     * \brief Method for getting the order of the kernel
     * 
     * \returns int  Order of the kernel 
     */
    inline int order() const;

    /*!
     * \brief Method for getting the value of the kernel at 0
     * 
     * \returns int Value of kernel at 0
     */
    inline int phiZero() const;

    /*!
     * \brief Method for evaluating the kernel for a given Distance
     * 
     * \param Distance  Distance for which to evaluate the kernel
     * 
     * \returns double Value of kernel at Distance
     */
    inline double eval(double Distance) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a given Distance 
     * 
     * \param Distance Distance for which to evaluate the derivative of the kernel
     * 
     * \returns double Derivative of kernel at Distance
     */
    inline double deriv(double Distance) const;

    /*!
     * \brief Method for evaluating the kernel for a matrix of distances
     * 
     * \param dists  Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EMatrixXd Values of kernel at dists
     */
    inline EMatrixXd eval(const EMatrixXd &dists) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a matrix of distances
     * 
     * \param dists Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EMatrixXd Values of the derivative of the kernel at dists
     */
    inline EMatrixXd deriv(const EMatrixXd &dists) const;

  private:
    /*! Value of the cubic kernel at 0 */
    int mPhiZero;

    /*! Order of the cubic kernel */
    int mOrder;
};

cubicKernel::cubicKernel() : mPhiZero(0), mOrder(2) {}

cubicKernel::~cubicKernel() {}

inline int cubicKernel::order() const
{
    return mOrder;
}

inline int cubicKernel::phiZero() const
{
    return mPhiZero;
}

inline double cubicKernel::eval(double Distance) const
{
    return Distance * Distance * Distance;
}

inline double cubicKernel::deriv(double Distance) const
{
    return 3 * Distance * Distance;
}

inline EMatrixXd cubicKernel::eval(const EMatrixXd &dists) const
{
    return dists.cwiseProduct(dists.cwiseProduct(dists));
}

inline EMatrixXd cubicKernel::deriv(const EMatrixXd &dists) const
{
    return 3 * dists.cwiseProduct(dists);
}

/*! \class thinPlateKernel
 * \ingroup Surrogate_Module
 * 
 * \brief Thin-plate spline kernel 
 * 
 * This is an implementation of the popular thin-plate spline kernel \f$\varphi(r)=r^2\,\log(r)\f$ which is of order 2.
 *
 * \author David Eriksson, dme65@cornell.edu
 */
class thinPlateKernel : public radialBasisFunctionKernel
{
  public:
    /*!
     * \brief Construct a new thin Plate Kernel object
     * 
     */
    thinPlateKernel();

    /*!
     * \brief Destroy the thin Plate Kernel object
     * 
     */
    ~thinPlateKernel();

    /*!
     * \brief Method for getting the order of the kernel
     * 
     * \returns int  Order of the kernel 
     */
    inline int order() const;

    /*!
     * \brief Method for getting the value of the kernel at 0
     * 
     * \returns int Value of kernel at 0
     */
    inline int phiZero() const;

    /*!
     * \brief Method for evaluating the kernel for a given Distance
     * 
     * \param Distance  Distance for which to evaluate the kernel
     * 
     * \returns double Value of kernel at Distance
     */
    inline double eval(double Distance) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a given Distance 
     * 
     * \param Distance Distance for which to evaluate the derivative of the kernel
     * 
     * \returns double Derivative of kernel at Distance
     */
    inline double deriv(double Distance) const;

    /*!
     * \brief Method for evaluating the kernel for a matrix of distances
     * 
     * \param dists  Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EMatrixXd Values of kernel at dists
     */
    inline EMatrixXd eval(const EMatrixXd &dists) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a matrix of distances
     * 
     * \param dists Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EMatrixXd Values of the derivative of the kernel at dists
     */
    inline EMatrixXd deriv(const EMatrixXd &dists) const;

  private:
    /*! Value of the cubic kernel at 0 */
    int mPhiZero;

    /*! Order of the cubic kernel */
    int mOrder;
};

thinPlateKernel::thinPlateKernel() : mPhiZero(0), mOrder(2) {}

thinPlateKernel::~thinPlateKernel() {}

inline int thinPlateKernel::order() const
{
    return mOrder;
}

inline int thinPlateKernel::phiZero() const
{
    return mPhiZero;
}

inline double thinPlateKernel::eval(double Distance) const
{
    return Distance * Distance * std::log(Distance + 1e-12);
}

inline double thinPlateKernel::deriv(double Distance) const
{
    return Distance * (1.0 + 2.0 * std::log(Distance + 1e-12));
}

inline EMatrixXd thinPlateKernel::eval(const EMatrixXd &dists) const
{
    return dists.cwiseProduct(dists.cwiseProduct((dists.array() + 1e-12).log().matrix()));
}

inline EMatrixXd thinPlateKernel::deriv(const EMatrixXd &dists) const
{
    return dists.cwiseProduct((1 + 2.0 * (dists.array() + 1e-12).log()).matrix());
}

} // namespace umuq

#endif // UMUQ_RADIALBASISFUNCTIONKERNEL

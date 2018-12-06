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
     * \returns double Value of kernel at 0
     */
    virtual inline double phiZero() const = 0;

    /*!
     * \brief Method for evaluating the kernel for a given Distance
     * 
     * \param Distance  Distance for which to evaluate the kernel
     * 
     * \returns double Value of kernel at Distance
     */
    virtual inline double evaluate(double const Distance) const = 0;

    /*!
     * \brief Method for evaluating the kernel for a vector of distances
     * 
     * \param Distance  Distances for which to evaluate the kernel
     * 
     * \returns EVectorXd Values of kernel at Distance
     */
    virtual inline EVectorXd evaluate(EVectorXd const &Distance) const = 0;

    /*!
     * \brief Method for evaluating the kernel for a matrix of distances
     * 
     * \param Distance  Distances for which to evaluate the kernel
     * 
     * \returns EMatrixXd Values of kernel at Distance
     */
    virtual inline EMatrixXd evaluate(EMatrixXd const &Distance) const = 0;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a given Distance 
     * 
     * \param Distance Distance for which to evaluate the derivative of the kernel
     * 
     * \returns double Derivative of kernel at Distance
     */
    virtual inline double deriv(double const Distance) const = 0;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a vector of distances
     * 
     * \param Distance Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EVectorXd Values of the derivative of the kernel at Distance
     */
    virtual inline EVectorXd deriv(EVectorXd const &Distance) const = 0;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a matrix of distances
     * 
     * \param Distance Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EMatrixXd Values of the derivative of the kernel at Distance
     */
    virtual inline EMatrixXd deriv(EMatrixXd const &Distance) const = 0;
};

radialBasisFunctionKernel::radialBasisFunctionKernel() {}

radialBasisFunctionKernel::~radialBasisFunctionKernel() {}

/*! \class linearKernel
 * \ingroup Surrogate_Module
 *
 * \brief Linear kernel
 * 
 * This is an implementation of the linear kernel \f$\varphi(r)=r\,\log(r)\f$ which is of order 1.
 */
class linearKernel : public radialBasisFunctionKernel
{
  public:
    /*!
     * \brief Construct a new linearKernel object
     * 
     */
    linearKernel();

    /*!
     * \brief Destroy the linearKernel object
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
     * \returns double Value of kernel at 0
     */
    inline double phiZero() const;

    /*!
     * \brief Method for evaluating the kernel for a given Distance
     * 
     * \param Distance  Distance for which to evaluate the kernel
     * 
     * \returns double Value of kernel at Distance
     */
    inline double evaluate(double const Distance) const;

    /*!
     * \brief Method for evaluating the kernel for a vector of distances
     * 
     * \param Distance  Distances for which to evaluate the kernel
     * 
     * \returns EVectorXd Values of kernel at Distance
     */
    inline EVectorXd evaluate(EVectorXd const &Distance) const;

    /*!
     * \brief Method for evaluating the kernel for a matrix of distances
     * 
     * \param Distance  Distances for which to evaluate the kernel
     * 
     * \returns EMatrixXd Values of kernel at Distance
     */
    inline EMatrixXd evaluate(EMatrixXd const &Distance) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a given Distance 
     * 
     * \param Distance Distance for which to evaluate the derivative of the kernel
     * 
     * \returns double Derivative of kernel at Distance
     */
    inline double deriv(double const Distance) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a vector of distances
     * 
     * \param Distance Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EVectorXd Values of the derivative of the kernel at Distance
     */
    inline EVectorXd deriv(EVectorXd const &Distance) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a matrix of distances
     * 
     * \param Distance Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EMatrixXd Values of the derivative of the kernel at Distance
     */
    inline EMatrixXd deriv(EMatrixXd const &Distance) const;

  private:
    /*! Order of the Linear kernel */
    int kernelOrder;
};

linearKernel::linearKernel() : kernelOrder(1) {}

linearKernel::~linearKernel() {}

inline int linearKernel::order() const { return kernelOrder; }

inline double linearKernel::phiZero() const { return 0; }

inline double linearKernel::evaluate(double const Distance) const { return Distance; }

inline EVectorXd linearKernel::evaluate(EVectorXd const &Distance) const { return Distance; }

inline EMatrixXd linearKernel::evaluate(EMatrixXd const &Distance) const { return Distance; }

inline double linearKernel::deriv(double const Distance) const { return 1.0; }

inline EVectorXd linearKernel::deriv(EVectorXd const &Distance) const
{
    return EVectorXd::Ones(Distance.rows());
}

inline EMatrixXd linearKernel::deriv(EMatrixXd const &Distance) const
{
    return EMatrixXd::Ones(Distance.rows(), Distance.cols());
}

/*! \class cubicKernel
 * \ingroup Surrogate_Module
 *
 * \brief Cubic kernel for a radial basis function
 *
 * This is an implementation of the popular cubic kernel \f$\varphi(r)=r^3\f$ which is of order 2.
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
     * \returns double Value of kernel at 0
     */
    inline double phiZero() const;

    /*!
     * \brief Method for evaluating the kernel for a given Distance
     * 
     * \param Distance  Distance for which to evaluate the kernel
     * 
     * \returns double Value of kernel at Distance
     */
    inline double evaluate(double const Distance) const;
    
    /*!
     * \brief Method for evaluating the kernel for a vector of distances
     * 
     * \param Distance  Distances for which to evaluate the kernel
     * 
     * \returns EVectorXd Values of kernel at Distance
     */
    inline EVectorXd evaluate(EVectorXd const &Distance) const;

    /*!
     * \brief Method for evaluating the kernel for a matrix of distances
     * 
     * \param Distance  Distances for which to evaluate the kernel
     * 
     * \returns EMatrixXd Values of kernel at Distance
     */
    inline EMatrixXd evaluate(EMatrixXd const &Distance) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a given Distance 
     * 
     * \param Distance Distance for which to evaluate the derivative of the kernel
     * 
     * \returns double Derivative of kernel at Distance
     */
    inline double deriv(double const Distance) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a vector of distances
     * 
     * \param Distance Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EVectorXd Values of the derivative of the kernel at Distance
     */
    inline EVectorXd deriv(EVectorXd const &Distance) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a matrix of distances
     * 
     * \param Distance Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EMatrixXd Values of the derivative of the kernel at Distance
     */
    inline EMatrixXd deriv(EMatrixXd const &Distance) const;

  private:
    /*! Order of the cubic kernel */
    int kernelOrder;
};

cubicKernel::cubicKernel() : kernelOrder(2) {}

cubicKernel::~cubicKernel() {}

inline int cubicKernel::order() const { return kernelOrder; }

inline double cubicKernel::phiZero() const { return 0; }

inline double cubicKernel::evaluate(double const Distance) const { return Distance * Distance * Distance; }

inline EVectorXd cubicKernel::evaluate(EVectorXd const &Distance) const
{
    return Distance.cwiseProduct(Distance.cwiseProduct(Distance)).matrix();
}

inline EMatrixXd cubicKernel::evaluate(EMatrixXd const &Distance) const
{
    return Distance.cwiseProduct(Distance.cwiseProduct(Distance)).matrix();
}

inline double cubicKernel::deriv(double const Distance) const { return 3 * Distance * Distance; }

inline EVectorXd cubicKernel::deriv(EVectorXd const &Distance) const
{
    return 3 * Distance.cwiseProduct(Distance).matrix();
}

inline EMatrixXd cubicKernel::deriv(EMatrixXd const &Distance) const
{
    return 3 * Distance.cwiseProduct(Distance).matrix();
}

/*! \class thinPlateKernel
 * \ingroup Surrogate_Module
 * 
 * \brief Thin-plate spline kernel 
 * 
 * This is an implementation of the popular thin-plate spline kernel \f$\varphi(r)=r^2\,\log(r)\f$ which is of order 2.
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
     * \returns int Order of the kernel 
     */
    inline int order() const;

    /*!
     * \brief Method for getting the value of the kernel at 0
     * 
     * \returns double Value of kernel at 0
     */
    inline double phiZero() const;

    /*!
     * \brief Method for evaluating the kernel for a given Distance
     * 
     * \param Distance  Distance for which to evaluate the kernel
     * 
     * \returns double Value of kernel at Distance
     */
    inline double evaluate(double const Distance) const;

    /*!
     * \brief Method for evaluating the kernel for a matrix of distances
     * 
     * \param Distance  Distances for which to evaluate the kernel
     * 
     * \returns EVectorXd Values of kernel at Distance
     */
    inline EVectorXd evaluate(EVectorXd const &Distance) const;

    /*!
     * \brief Method for evaluating the kernel for a matrix of distances
     * 
     * \param Distance  Distances for which to evaluate the kernel
     * 
     * \returns EMatrixXd Values of kernel at Distance
     */
    inline EMatrixXd evaluate(EMatrixXd const &Distance) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a given Distance 
     * 
     * \param Distance Distance for which to evaluate the derivative of the kernel
     * 
     * \returns double Derivative of kernel at Distance
     */
    inline double deriv(double const Distance) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a matrix of distances
     * 
     * \param Distance Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EVectorXd Values of the derivative of the kernel at Distance
     */
    inline EVectorXd deriv(EVectorXd const &Distance) const;

    /*!
     * \brief Method for evaluating the derivative of the kernel for a matrix of distances
     * 
     * \param Distance Distances for which to evaluate the derivative of the kernel
     * 
     * \returns EMatrixXd Values of the derivative of the kernel at Distance
     */
    inline EMatrixXd deriv(EMatrixXd const &Distance) const;

  private:
    /*! Order of the Thin-plate spline kernel */
    int kernelOrder;
};

thinPlateKernel::thinPlateKernel() : kernelOrder(2) {}

thinPlateKernel::~thinPlateKernel() {}

inline int thinPlateKernel::order() const { return kernelOrder; }

inline double thinPlateKernel::phiZero() const { return 0; }

inline double thinPlateKernel::evaluate(double const Distance) const
{
    return Distance * Distance * std::log(Distance + machinePrecision<double>);
}

inline EVectorXd thinPlateKernel::evaluate(EVectorXd const &Distance) const
{
    return Distance.cwiseProduct(Distance.cwiseProduct((Distance.array() + machinePrecision<double>).log().matrix())).matrix();
}

inline EMatrixXd thinPlateKernel::evaluate(EMatrixXd const &Distance) const
{
    return Distance.cwiseProduct(Distance.cwiseProduct((Distance.array() + machinePrecision<double>).log().matrix())).matrix();
}

inline double thinPlateKernel::deriv(double const Distance) const
{
    return Distance * (1.0 + 2.0 * std::log(Distance + machinePrecision<double>));
}

inline EVectorXd thinPlateKernel::deriv(EVectorXd const &Distance) const
{
    return Distance.cwiseProduct((1 + 2.0 * (Distance.array() + machinePrecision<double>).log()).matrix()).matrix();
}

inline EMatrixXd thinPlateKernel::deriv(EMatrixXd const &Distance) const
{
    return Distance.cwiseProduct((1 + 2.0 * (Distance.array() + machinePrecision<double>).log()).matrix()).matrix();
}

} // namespace umuq

#endif // UMUQ_RADIALBASISFUNCTIONKERNEL

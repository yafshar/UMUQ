#ifndef UMUQ_SURROGATE_H
#define UMUQ_SURROGATE_H

#include "numerics/eigenlib.hpp"

namespace umuq
{

/*! \file surrogate.hpp
 * \ingroup 
 * 
 * \brief Implementation of the Radial basis function kernel.
 *
 * \author David Eriksson, dme65@cornell.edu
 * 
 * This file contains minor addition to the original surrogate.h
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

/*! 
 * \defgroup Surrogate_Module Surrogate module
 * This is the surrogate module of %UMUQ providing all necessary classes for using surrogates of models.
 * These surrogates map an \f$ n\f$ dimensional parameter space to the real values. 
 * That is  \f$ f: \mathbb{R}^n \rightarrow \mathbb{R} \f$. 
 */






/*! \class surrogate
 * \ingroup Surrogate_Module
 *
 * \brief Abstract class for a surrogate model
 * 
 * This is the abstract class that should be used as a Base class for all
 * surrogate models.
 */
class surrogate
{
public:
  /*!
   * \brief Construct a new surrogate object
   * 
   */
  surrogate();

  /*!
   * \brief Destroy the surrogate object
   * 
   */
  ~surrogate();

  /*!
   * \brief Method for getting the current number of points
   * 
   * \returns int Current number of points
   */
  virtual int numPoints() const = 0;

  /*!
   * \brief Get the Number of Dimensions 
   * 
   * \returns int Number of dimensions
   */
  virtual int getNumDimensions() const = 0;

  /*!
   * \brief Method for resetting the surrogate model
   * 
   */
  virtual void reset() = 0;

  /*!
   * \brief Method for getting the current points
   * 
   * \returns EMatrixXd Current points
   */
  virtual EMatrixXd getCurrentPoints() const = 0;

  /*!
   * \brief Method for getting current point number i (0 is the first)
   * 
   * \param i Index number
   * 
   * \returns EVectorXd Point at index number i
   */
  virtual EVectorXd getCurrentPoints(int i) const = 0;

  /*!
   * \brief Method for getting the values of the current points
   * 
   * \returns EVectorXd Values of current points 
   */
  virtual EVectorXd getFunctionValues() const = 0;

  /*!
   * \brief Method for getting the value of current point number i (0 is the first)
   * 
   * \param i Index number
   * 
   * \returns double Value of point at index number i 
   */
  virtual double getFunctionValues(int i) const = 0;

  /*!
   * \brief Method for adding a point with a known value
   * 
   * \param Point          Point to be added
   * \param FunctionValue  Function value at point
   */
  virtual void addPoint(EVectorXd const &Point, double FunctionValue) = 0;

  /*!
   * \brief  Method for adding multiple points with known values
   * 
   * \param Points          Points to be added
   * \param FunctionValues  Function values at the points
   */
  virtual void addPoints(EMatrixXd const &Points, EVectorXd const &FunctionValues) = 0;

  /*!
   * \brief Method for evaluating the surrogate model at a point
   * 
   * \param Point Point for which to evaluate the surrogate
   * 
   * \returns double Value of the surrogate model at the point
   */
  virtual double evaluate(EVectorXd const &Point) const = 0;

  /*!
   * \brief Method for evaluating the surrogate at multiple points
   * 
   * \param Point     Point for which to evaluate the surrogate model
   * \param Distance  Distances between the interpolation nodes and point
   * 
   * \returns double Values of the surrogate model at the points
   */
  virtual double evaluate(EVectorXd const &Point, EVectorXd const &Distance) const = 0;

  /*!
   * \brief Method for evaluating the surrogate at multiple points
   * 
   * \param Points Points for which to evaluate the surrogate model
   * 
   * \returns double Values of the surrogate model at the points
   */
  virtual EVectorXd evaluate(EMatrixXd const &Points) const = 0;

  /*!
   * \brief Method for evaluating the surrogate at multiple points
   * 
   * \param Points     Points for which to evaluate the surrogate model
   * \param Distances  Distances between the interpolation nodes and the points
   * 
   * \returns EVectorXd Values of the surrogate model at the points
   */
  virtual EVectorXd evaluate(EMatrixXd const &Points, EMatrixXd const &Distances) const = 0;

  /*!
   * \brief Method for evaluating the derivative of the surrogate model at a point
   * 
   * \param Point  Point for which to evaluate the surrogate model
   * 
   * \returns EVectorXd Value of the derivative of the surrogate model at the points
   */
  virtual EVectorXd deriv(EVectorXd const &Point) const = 0;

  /*!
   * \brief Method for fitting the surrogate model
   * 
   */
  virtual void fit() = 0;
};

surrogate::surrogate() {}

surrogate::~surrogate() {}

} // namespace umuq

#endif // UMUQ_SURROGATE

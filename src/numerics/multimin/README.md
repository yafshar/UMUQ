Multidimensional Minimization Module
------------

The multimin Module contains the c++ re-implementation and modifications to the original<br>
GSL Multidimensional Minimization source codes made available under the following license:

~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Copyright (C) 1996, 1997, 1998, 1999, 2000 Fabrice Rossi
 
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3 of the License, or (at
 your option) any later version.
 
 This program is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contents
----------------

It should contain the following files:  

````
    multimin
    |-- COPYING.GPL
    |-- bfgs2.hpp
    |-- bfgs.hpp
    |-- conjugatefr.hpp
    |-- conjugatepr.hpp
    |-- simplexnm2.hpp
    |-- simplexnm2rnd.hpp
    |-- simplexnm.hpp
    `-- steepestdescent.hpp
````

For local optimization, the most efficient algorithms typically require the user to supply 
the gradient in addition to the value f(x) for any given point x. This exploits the fact 
that, in principle, the gradient can almost always be computed at the same time as the value 
of f using very little additional computational effort (at worst, about the same as that of 
evaluating f a second time). If a quick way to compute the derivative of f is not obvious, 
one typically finds gradient using an adjoint method, or possibly using automatic differentiation 
tools.<br>
Gradient-based methods are critical for the efficient optimization of very high-dimensional 
parameter spaces (e.g. n in the thousands or more).
On the other hand, computing the gradient is sometimes cumbersome and inconvenient if the 
objective function is supplied as a complicated program. It may even be impossible, if f 
is not differentiable (or worse, is discontinuous). In such cases, it is often easier to 
use a derivative-free algorithm for optimization, which only requires that the user supply 
the function values f(x) for any given point x. Such methods typically must evaluate f for 
at least several-times-n points, however, so they are best used when n is small to moderate 
(up to hundreds).

The available solvers can be categorized as follows:

**Algorithms with Derivatives**

Solver | Description
--- | --- 
bfgs | Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm
bfgs2 | Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm (the most efficient version)
conjugateFr | Conjugate gradient Fletcher-Reeve algorithm
conjugatePr | Conjugate gradient Polak-Ribiere algorithm
steepestDescent | The steepest descent algorithm

**Algorithms without Derivatives**

Solver | Description 
--- | ---|
simplexNM | The Simplex method of Nelder and Mead
simplexNM2 | The Simplex method of Nelder and Mead$ (order N operations)
simplexNM2Rnd | The Simplex method of Nelder and Mead (Uses a randomly-oriented set of basis vectors)

Contributors
------------
UMUQ package maintainer: Yaser Afshar <yafshar@umich.edu>  

Computational Aerosciences Laboratory<br>
University of Michigan, Ann Arbor

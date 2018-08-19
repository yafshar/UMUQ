# UMUQ

Multidimensional Minimization
------------

The multimin Module contains the c++ re-implamentation and modifications to the original 
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

-----------------------------------
    multimin
    ├── COPYING
    ├── bfgs.hpp
    ├── bfgs2.hpp
    ├── conjugatefr.hpp
    ├── conjugatepr.hpp
    ├── simplexnm.hpp
    ├── simplexnm2.hpp
    ├── simplexnm2rnd.hpp
    ├── steepestdescent.hpp
    └── README.md
-----------------------------------


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

Computational Aerosciences Laboratory  
University of Michigan, Ann Arbor 

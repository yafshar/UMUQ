Function Module
----------------

The Module contains all the function classes.

Contents
----------------

It should contain the following files:  

````
    function
    |-- densityfunction.hpp
    |-- differentiablefunctionminimizer.hpp
    |-- fitfunction.hpp
    |-- functionminimizer.hpp
    |-- linearfunctionwrapper.hpp
    |-- umuqdifferentiablefunction.hpp
    |-- umuqfunction.hpp
    `-- utilityfunction.hpp
````

Function | Description
:--- | :---
densityFunction                 | A density function or a probability density (PDF) function class
differentiableFunctionMinimizer | A base class which is for finding minima of arbitrary multidimensional functions with derivative
fitFunction                     | A base class which is for fitting function in the inference process
functionMinimizer               | A base class which is for finding minima of arbitrary multidimensional functions
linearFunctionWrapper           | Wrapper for an external Multidimensional function
umuqDifferentiableFunction      | A general-purpose polymorphic differentiable function wrapper of n variables
umuqFunction                    | A general-purpose polymorphic function wrapper of n variables
utilityfunction                 | Helper functions

Contributors
----------------
UMUQ package maintainer: Yaser Afshar <yafshar@umich.edu>  

Computational Aerosciences Laboratory<br>
University of Michigan, Ann Arbor

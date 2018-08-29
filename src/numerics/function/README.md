# UMUQ

Function Module
------------

The Module contains all the function classes 

Contents
----------------

It should contain the following files:  

-----------------------------------
    multimin
    ├── densityfunction.hpp
    ├── differentiablefunctionminimizer.hpp
    ├── fitfunction.hpp
    ├── functionminimizer.hpp
    ├── functiontype.hpp
    ├── linearfunctionwrapper.hpp
    ├── umuqdifferentiablefunction.hpp
    ├── umuqfunction.hpp
    ├── utilityfunction.hpp
    └── README.md
-----------------------------------

**densityFunction** a density function or a probability density (PDF) function class    
**differentiableFunctionMinimizer** a base class which is for finding minima of arbitrary multidimensional functions with derivative    
**fitFunction**  a base class which is for fitting function in the inference process     
**functionMinimizer** a base class which is for finding minima of arbitrary multidimensional functions    
**functiontype** a collection of Function types for convenience use    
**linearFunctionWrapper** wrapper for an external Multidimensional function    
**umuqDifferentiableFunction** a general-purpose polymorphic differentiable function wrapper of n variables    
**umuqFunction** a general-purpose polymorphic function wrapper of n variables    
**utilityfunction** helper functions    

Contributors
------------
UMUQ package maintainer: Yaser Afshar <yafshar@umich.edu>  

Computational Aerosciences Laboratory  
University of Michigan, Ann Arbor 

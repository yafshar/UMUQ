# UMUQ

Random Module
------------

The Module contains all the PRNG classes currenly implemented in UMUQ    

It contains Engines and distributions used to produce random values. 
Also, the psrandom class generates pseudo-random numbers. 
All of the engines may be specifically seeded, for use with repeatable simulators. 
Random number engines generate pseudo-random numbers using seed data as entropy source. 
The choice of which engine to use involves a number of tradeoffs:    
 
Saru PRNG has only a small storage requirement for state which is 64-bit and is very fast.    

The Mersenne twister is slower and has greater state storage requirements but with the right parameters has 
the longest non-repeating sequence with the most desirable spectral characteristics (for a given definition of desirable).    

Contents
----------------

It should contain the following files:    

-----------------------------------
    random
    ├── multinomial.hpp
    ├── psrandom.hpp
    └── saruprng.hpp
-----------------------------------

Contributors       
------------
UMUQ package maintainer: Yaser Afshar <yafshar@umich.edu>  

Computational Aerosciences Laboratory  
University of Michigan, Ann Arbor 
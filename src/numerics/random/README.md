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
    ├── psrandom_exponentialdistribution.hpp
    ├── psrandom_gammadistribution.hpp
    ├── psrandom.hpp
    ├── psrandom_lognormaldistribution.hpp
    ├── psrandom_multinomial.hpp
    ├── psrandom_multivariatenormaldistribution.hpp
    ├── psrandom_normaldistribution.hpp
    ├── README.md
    └── saruprng.hpp
-----------------------------------

Random number distributions
----------------

A random number distribution post-processes the output of a PRNGs in such a way that resulting output 
is distributed according to a defined statistical probability density function. 

Currently umuq has some classes of random number distributions which include:
**normalDistribution**             produces real values on a standard normal (Gaussian) distribution    
**NormalDistribution**             produces real values on a standard normal (Gaussian) distribution    
**lognormalDistribution**          produces real values on a lognormal distribution   
**logNormalDistribution**          produces real values on a lognormal distribution    
**gammaDistribution**              produces real values on an gamma distribution   
**exponentialDistribution**        produces real values on an exponential distribution    
**multivariatenormalDistribution** produces real values on a multivariate normal (Gaussian, or joint normal) distribution      
**multivariateNormalDistribution** produces real values on a multivariate normal (Gaussian, or joint normal) distribution    

Contributors       
------------
UMUQ package maintainer: Yaser Afshar <yafshar@umich.edu>  

Computational Aerosciences Laboratory  
University of Michigan, Ann Arbor 
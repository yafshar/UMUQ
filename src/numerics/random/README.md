Random Module
------------

The Module contains all the PRNG classes currently implemented in UMUQ

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

````
    random
    |-- psrandom_exponentialdistribution.hpp
    |-- psrandom_gammadistribution.hpp
    |-- psrandom.hpp
    |-- psrandom_lognormaldistribution.hpp
    |-- psrandom_multinomial.hpp
    |-- psrandom_multivariatenormaldistribution.hpp
    |-- psrandom_normaldistribution.hpp
    `-- saruprng.hpp
````

Random number distributions
----------------

A random number distribution post-processes the output of a PRNGs in such a way that resulting output 
is distributed according to a defined statistical probability density function.

**Random number distributions currently supported in UMUQ**

Distributions | Description
:--- | :--- 
[normalDistribution](https://en.wikipedia.org/wiki/Normal_distribution)                          | produces real values on a standard normal (Gaussian) distribution
[lognormalDistribution](https://en.wikipedia.org/wiki/Log-normal_distribution)                   | produces real values on a lognormal distribution
[gammaDistribution](https://en.wikipedia.org/wiki/Gamma_distribution)                            | produces real values on an gamma distribution
[exponentialDistribution](https://en.wikipedia.org/wiki/Exponential_distribution)                | produces real values on an exponential distribution
[multivariatenormalDistribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) | produces real values on a multivariate normal (Gaussian, or joint normal) distribution
[multinomialDistribution](https://en.wikipedia.org/wiki/Multinomial_distribution)                | produces the probability of sampling n[K] from a multinomial distribution with parameters p[K], using:  <img src="https://latex.codecogs.com/svg.latex?&space;Pr(X_1=n_1,%20\cdots,%20X_K=n_K)%20=%20\frac{N!}{\left(n_1!%20n_2!%20\cdots%20n_K!%20\right)}%20p_1^{n_1}%20p_2^{n_2}%20\cdots%20p_K^{n_K}" title="multinomial" /> </td>


Contributors
------------
UMUQ package maintainer: Yaser Afshar <yafshar@umich.edu>  

Computational Aerosciences Laboratory<br>
University of Michigan, Ann Arbor
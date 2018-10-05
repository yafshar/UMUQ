Density Function Module
------------

The Module contains all the density function classes.

A density function or a probability density (PDF), is a function, with a value at any given point (or sample point) 
interpreted as a relative likelihood that the value of the random variable would be equal to that sample.
The value of the PDF at two different samples can be used to infer, in any particular draw of the random variable, 
how much more likely it is that the random variable would equal one sample compared to the other sample.

Reference:
[https://en.wikipedia.org/wiki/Probability_density_function](https://en.wikipedia.org/wiki/Probability_density_function)

Contents
----------------

It should contain the following files:

````
    density
    |-- exponentialdistribution.hpp
    |-- gammadistribution.hpp
    |-- gaussiandistribution.hpp
    |-- multinomialdistribution.hpp
    |-- multivariategaussiandistribution.hpp
    `-- uniformdistribution.hpp
````

Contributors
------------
UMUQ package maintainer: Yaser Afshar <yafshar@umich.edu>  

Computational Aerosciences Laboratory<br>
University of Michigan, Ann Arbor

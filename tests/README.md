Tests 
------------
We are using [**Google Test**, Google's C++ test framework](https://github.com/google/googletest) as our unit testing framework.<br>
If you are not familliar with Google test, please read the [introduction to Google C++ Testing Framework](https://github.com/google/googletest/blob/master/googletest/docs/primer.md) and learn how to write tests using Google Test.

Tests contains unit tests for individual units of **UMUQ** source code.

Contents
----------------

It should contain the following directories:  

````
    tests
    |-- data
    |-- inference
    |-- io
    |-- misc
    |-- numerics
    `-- torc
````

and the following files:

````
    tests
    |-- data
    |   |-- database_test.cpp
    |   |-- datatype_test.cpp
    |   |-- runinfo_test.cpp
    |   |-- stdata_test.cpp
    |   `-- test.txt
    |-- inference
    |   |-- prior
    |   |   `-- priordistribution_test.cpp
    |   `-- tmcmc
    |       `-- tmcmc_test.cpp
    |-- io
    |   |-- eigen_io_test.cpp
    |   |-- io_test.cpp
    |   `-- pyplot_test.cpp
    |-- misc
    |   |-- arraywrapper_test.cpp
    |   |-- funcallcounter_test.cpp
    |   |-- parser_test.cpp
    |   |-- timer_test.cpp
    |   `-- utility_test.cpp
    |-- numerics
    |   |-- dcpse_test.cpp
    |   |-- density
    |   |   `-- densityfunction_test.cpp
    |   |-- eigen_test.cpp
    |   |-- factorial_test.cpp
    |   |-- fitness_test.cpp
    |   |-- function
    |   |   |-- data.txt
    |   |   |-- fitfunction_test.cpp
    |   |   |-- functionminimizer_test.cpp
    |   |   `-- function_test.cpp
    |   |-- knearestneighbors_test.cpp
    |   |-- knearestneighbors_test.txt
    |   |-- linearregression_test.cpp
    |   |-- multimin
    |   |   |-- bfgs2_test.cpp
    |   |   |-- bfgs_test.cpp
    |   |   |-- conjugatefr_test.cpp
    |   |   |-- conjugatepr_test.cpp
    |   |   |-- multimin_test.cpp
    |   |   |-- simplexnm2rnd_test.cpp
    |   |   |-- simplexnm2_test.cpp
    |   |   |-- simplexnm_test.cpp
    |   |   `-- steepestdescent_test.cpp
    |   |-- multimin_test.cpp
    |   |-- polynomials
    |   |   |-- legendrepolynomial_test.cpp
    |   |   `-- polynomial_test.cpp
    |   |-- random
    |   |   `-- psrandom_test.cpp
    |   `-- stats_test.cpp
    `-- torc
        `-- torc_test.cpp
````

Contributors
------------
UMUQ package maintainer: Yaser Afshar <yafshar@umich.edu>  

Computational Aerosciences Laboratory<br>
University of Michigan, Ann Arbor

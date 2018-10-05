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
    |   `-- tmcmc
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
    |   |-- eigen_test.cpp
    |   |-- factorial_test.cpp
    |   |-- fitness_test.cpp
    |   |-- function
    |   |-- knearestneighbors_test.cpp
    |   |-- knearestneighbors_test.txt
    |   |-- linearregression_test.cpp
    |   |-- multimin
    |   |-- multimin_test.cpp
    |   |-- polynomial_test.cpp
    |   |-- random
    |   `-- stats_test.cpp
    `-- torc
        `-- torc_test.cpp
````

Contributors
------------
UMUQ package maintainer: Yaser Afshar <yafshar@umich.edu>  

Computational Aerosciences Laboratory<br>
University of Michigan, Ann Arbor

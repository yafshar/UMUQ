# UMUQ

Tests 
------------
We are using [**Google Test**, Google's C++ test framework](https://github.com/google/googletest) as our unit testing framework.
If you are not familliar with Google test, please read the [introduction to Google C++ Testing Framework](https://github.com/google/googletest/blob/master/googletest/docs/primer.md) and learn how to write tests using Google Test.    

Tests contains unit tests for individual units of **UMUQ** source code.

Contents
----------------

It should contain the following directories:  

-----------------------------------
    tests
    ├── data
    ├── io
    ├── misc
    └── numerics
        ├── densityfunction
        └── random
-----------------------------------  


and the following files:    

-----------------------------------
    tests
    ├── Makefile.am
    ├── Makefile.in
    ├── README.md
    ├── data
    │   ├── database_test.cpp
    │   ├── datatype_test.cpp
    │   ├── runinfo_test.cpp
    │   ├── stdata_test.cpp
    │   └── test.txt
    ├── io
    │   ├── eigen_io_test.cpp
    │   ├── io_test.cpp
    │   └── pyplot_test.cpp
    ├── misc
    │   ├── array_test.cpp
    │   ├── funcallcounter_test.cpp
    │   ├── parser_test.cpp
    │   └── utility_test.cpp
    └── numerics
        ├── dcpse_test.cpp
        ├── densityfunction
        │   └── densityfunction_test.cpp
        ├── eigen_test.cpp
        ├── factorial_test.cpp
        ├── fitness_test.cpp
        ├── knearestneighbors_test.cpp
        ├── knearestneighbors_test.txt
        ├── linearregression_test.cpp
        ├── multimin_test.cpp
        ├── polynomial_test.cpp
        ├── random
        │   └── psrandom_test.cpp
        └── stats_test.cpp
-----------------------------------

Contributors       
------------
UMUQ package maintainer: Yaser Afshar <yafshar@umich.edu>  

Computational Aerosciences Laboratory  
University of Michigan, Ann Arbor 

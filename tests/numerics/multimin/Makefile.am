TESTS=
check_PROGRAMS=

AM_CPPFLAGS = -I$(top_srcdir)/src $(GTEST_CPPFLAGS)
AM_CXXFLAGS = -DUMUQ_UNITTEST $(GTEST_CXXFLAGS)
AM_LDFLAGS = $(GTEST_LDFLAGS)
LIBS = $(GTEST_LIBS)

#numerics/multimin
TESTS += conjugatefr_test
check_PROGRAMS += conjugatefr_test
conjugatefr_test_SOURCES = conjugatefr_test.cpp

TESTS += conjugatepr_test
check_PROGRAMS += conjugatepr_test
conjugatepr_test_SOURCES = conjugatepr_test.cpp

TESTS += simplexnm_test
check_PROGRAMS += simplexnm_test
simplexnm_test_SOURCES = simplexnm_test.cpp

TESTS += simplexnm2_test
check_PROGRAMS += simplexnm2_test
simplexnm2_test_SOURCES = simplexnm2_test.cpp

TESTS += simplexnm2rnd_test
check_PROGRAMS += simplexnm2rnd_test
simplexnm2rnd_test_SOURCES = simplexnm2rnd_test.cpp

TESTS += steepestdescent_test
check_PROGRAMS += steepestdescent_test
steepestdescent_test_SOURCES = steepestdescent_test.cpp

TESTS += bfgs_test
check_PROGRAMS += bfgs_test
bfgs_test_SOURCES = bfgs_test.cpp

TESTS += bfgs2_test
check_PROGRAMS += bfgs2_test
bfgs2_test_SOURCES = bfgs2_test.cpp

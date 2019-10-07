TESTS=
check_PROGRAMS=

AM_CPPFLAGS = -I$(top_srcdir)/src $(GTEST_CPPFLAGS)
AM_CXXFLAGS = -DUMUQ_UNITTEST $(GTEST_CXXFLAGS)
AM_LDFLAGS = $(GTEST_LDFLAGS)
LIBS = $(GTEST_LIBS)

#inference
#prior
TESTS += priordistribution_test
check_PROGRAMS += priordistribution_test
priordistribution_test_SOURCES = prior/priordistribution_test.cpp

#tmcmc
TESTS += stdata_test
check_PROGRAMS += stdata_test
stdata_test_SOURCES = tmcmc/stdata_test.cpp

TESTS += runinfo_test
check_PROGRAMS += runinfo_test
runinfo_test_SOURCES = tmcmc/runinfo_test.cpp

TESTS += database_test
check_PROGRAMS += database_test
database_test_SOURCES = tmcmc/database_test.cpp

TESTS += datatype_test
check_PROGRAMS += datatype_test
datatype_test_SOURCES = tmcmc/datatype_test.cpp

TESTS += tmcmc_test
check_PROGRAMS += tmcmc_test
tmcmc_test_SOURCES = tmcmc/tmcmc_test.cpp

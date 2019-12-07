TESTS=
check_PROGRAMS=

AM_CPPFLAGS = -I$(top_srcdir)/src $(GTEST_CPPFLAGS)
AM_CXXFLAGS = -DUMUQ_UNITTEST $(GTEST_CXXFLAGS)
AM_LDFLAGS = $(GTEST_LDFLAGS)
LIBS = $(GTEST_LIBS)

clean-local:
	rm -fr COORDS* FORCE* XCOORDS*

#3rdparty interface
#interface
TESTS += dftfe_test
check_PROGRAMS += dftfe_test
dftfe_test_SOURCES = dftfe_test.cpp

#python interface
TESTS += python_test
check_PROGRAMS += python_test
python_test_SOURCES = python_test.cpp

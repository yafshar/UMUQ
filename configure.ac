# University of Michigan's Uncertainty Quantification framework
#
# AUTHOR
#	Yaser Afshar @ ya.afshar@gmail.com
#	Dept of Aerospace Engineering | University of Michigan
#
AC_PREREQ([2.69])

# Given a version number MAJOR.MINOR.PATCH, increment the:
# MAJOR version when you make incompatible API changes,
# MINOR version when you add functionality in a backwards-compatible manner, and
# PATCH version when you make backwards-compatible bug fixes.
AC_INIT([UMUQ], [1.0.0], [ya.afshar@gmail.com])

AC_COPYRIGHT(Copyright 2018 University of Michigan)
AC_REVISION([$Revision: 1.0.0 $])

AC_CONFIG_AUX_DIR([auxdir])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_TESTDIR([tests])
AC_CONFIG_MACRO_DIR([m4])
# 
m4_include([m4/m4_ax_prog.m4])
m4_include([m4/m4_ax_arch.m4])
m4_include([m4/m4_ax_cxx_compile_stdcxx.m4])
m4_include([m4/m4_ax_mpi.m4])
m4_include([m4/m4_ax_pthread.m4])
m4_include([m4/m4_ax_torc.m4])
m4_include([m4/m4_ax_eigen.m4])
m4_include([m4/m4_ax_flann.m4])
m4_include([m4/m4_ax_googletest.m4])
m4_include([m4/m4_ax_doxygen.m4])
#

# Configuring
AC_MSG_RESULT(---------------------------------------------)
AC_MSG_RESULT(------------  Configuring UMUQ  -------------)
AC_MSG_RESULT(---------------------------------------------)

AC_CANONICAL_BUILD
AC_CANONICAL_HOST
AC_CANONICAL_TARGET
# Check for architectures
AX_ARCH

AC_CONFIG_HEADERS(include/UMUQ_config.h)

#
# Init automake
#
AM_INIT_AUTOMAKE([1.9 dist-bzip2 dist-zip foreign subdir-objects])

#
# Set prefix default (install directory) to a directory in the build area.
# This allows multiple src-dir builds within one host.
AC_PREFIX_DEFAULT([`pwd`])

# Checks for programs.
AX_PROG
LT_INIT

# Checks for libraries.
AC_MSG_RESULT() 

### Checking for MPI
AC_MSG_CHECKING([for - MPI Library - ])
AX_MPI

### Checking for POSIX threads
AC_MSG_CHECKING([for - POSIX threads - ])
AX_PTHREAD

### Checking for TORC
AC_MSG_CHECKING([for - TORC Library - ])
AX_TORC

### Checking for EIGEN
AC_MSG_CHECKING([for - EIGEN Library - ])
AX_EIGEN

### Checking for FLANN
AC_MSG_CHECKING([for - FLANN Library - ])
AX_FLANN

### Checking for GOOGLETEST
AC_MSG_CHECKING([for - GOOGLETEST Framework - ])
AX_GOOGLETEST

# Chech for Doxygen
AC_MSG_CHECKING([ for - DOXYGEN - ])
AX_DOXYGEN

AC_LANG_PUSH([C++])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE

# Ensure we are using c14 C++ standard.
AX_CXX_COMPILE_STDCXX([14], [noext], [mandatory])

# Checks for library functions.
AC_CHECK_FUNCS([pow])

AC_LANG_POP([C++])

AC_CONFIG_FILES([Makefile src/Makefile])
AM_COND_IF([HAVE_GOOGLETEST], [AC_CONFIG_FILES([tests/Makefile])])
AM_COND_IF([HAVE_DOXYGEN], [AC_CONFIG_FILES([docs/Makefile])])

AC_OUTPUT

AC_MSG_RESULT([
UMUQ framework $VERSION is now configured

Configure Information:
  C preprocessor    : $CPP
    CPPFLAGS        :   $CPPFLAGS

  C Compiler        : $CC
    DEFS            :   $DEFS
    CFLAGS          :   $CFLAGS

  C++ Compiler      : $CXX
    DEFS            :   $DEFS
    CXXFLAGS        :   $CXXFLAGS

  Linker            : $LD
    LDFLAGS         :   $LDFLAGS
    LIBS            :   $LIBS

  Google Test 
    GTEST_CPPFLAGS  : $GTEST_CPPFLAGS
    GTEST_CXXFLAGS  : $GTEST_CXXFLAGS
    GTEST_LDFLAGS   : $GTEST_LDFLAGS
    GTEST_LIBS      : $GTEST_LIBS
])

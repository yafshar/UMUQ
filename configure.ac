AC_PREREQ([2.69])
AC_INIT([DCPSE Surrogate], [1.0], [ya.afshar@gmail.com])
AC_CONFIG_SRCDIR([src/polynomial.h])
AM_INIT_AUTOMAKE

AC_MSG_RESULT(---------------------------------------------)
AC_MSG_RESULT(------- Configuring DCPSE Surrogate ---------)
AC_MSG_RESULT(---------------------------------------------)

AC_CONFIG_MACRO_DIR([./m4])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE

# --------------------------------------------------------------
# Standard autoconf macro for determining the proper -std=c++11
# flag for the current compiler.  Adds the result to CXXFLAGS if
# one is found.  See ax_cxx_compile_stdcxx_11.m4 for details.
# --------------------------------------------------------------
AX_CXX_COMPILE_STDCXX_11


# Checks for library functions.
AC_CHECK_FUNCS([pow])

AC_OUTPUT(Makefile src/Makefile)

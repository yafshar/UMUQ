#
#   AX_EIGEN([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   This macro looks for EIGEN library (see http://www.netlib.org/lapack/)
#   On success, it sets the EIGEN_INCLUDE output variable to hold the 
#   requisite includes.
#
#   The user may also use --with-eigen=<include> in order to use some specific
#   Eigen library.
#
#   ACTION-IF-FOUND is a list of shell commands to run if a Eigen library
#   is found, and ACTION-IF-NOT-FOUND is a list of commands to run it if it
#   is not found. If ACTION-IF-FOUND is not specified, the default action
#   will define HAVE_EIGEN.
#
# LICENSE
#
#   Copyright (c) 2009 Steven G. Johnson <stevenj@alum.mit.edu>
#
#   This program is free software: you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation, either version 3 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <http://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Archive. When you make and distribute a
#   modified version of the Autoconf Macro, you may extend this special
#   exception to the GPL to apply to your modified version as well.

#serial 1

AC_DEFUN([AX_EIGEN], [
        AC_ARG_WITH([eigen],
        AS_HELP_STRING([--with-eigen@<:@=DIR@:>@], [use EIGEN libraries (default is yes) - it is possible to specify the root directory for EIGEN (optional)]), 
        [ 
                if test "$withval" = "no"; then
                        want_eigen="no"
                elif test "$withval" = "yes"; then
                        want_eigen="yes"
                        ac_eigen_path=""
                else
                        want_eigen="yes"
                        ac_eigen_path="$withval"
                fi
        ], [want_eigen="yes"])
   
        if test "x$want_eigen" = "xyes"; then
                AC_MSG_CHECKING(for eigenlib header files)
                succeeded=no
                
                dnl first we check the system location for eigen libraries
                if test "$ac_eigen_path" != ""; then
                        EIGEN_CPPFLAGS="-I$ac_eigen_path/include -I$ac_eigen_path/include/eigen3"
                else
                        for ac_eigen_path_tmp in /usr /usr/local /opt /opt/local ; do
                                if test -d "$ac_eigen_path_tmp/include/eigen" && test -r "$ac_eigen_path_tmp/include/eigen"; then
                                        EIGEN_CPPFLAGS="-I$ac_eigen_path_tmp/include -I$ac_eigen_path_tmp/include/eigen3"
                                        break;
                                fi
                        done
                fi

                CPPFLAGS_SAVED="$CPPFLAGS"
                CPPFLAGS+="$EIGEN_CPPFLAGS"

                AC_LANG_PUSH(C++)
                AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                        @%:@include <Eigen/Dense>
                ]], [[ ]]
                )], [AC_MSG_RESULT(yes)
                succeeded=yes
                found_system=yes
                ],[])
                AC_LANG_POP([C++])

                if test "$succeeded" == "yes" ; then
                        AC_SUBST(CPPFLAGS)
                        AC_DEFINE(HAVE_EIGEN,,[define if the EIGEN library is available])
                        ax_eigen_ok="yes"
                else
                        CPPFLAGS="$CPPFLAGS_SAVED"
                fi
        fi
])
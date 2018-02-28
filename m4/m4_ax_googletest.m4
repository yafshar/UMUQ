#
# SYNOPSIS
#
#   AX_GOOGLETEST([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   Test for the GOOGLETEST framework
#
#   If no path to the installed googletest framework is given the macro searchs
#   under /usr, /usr/local, /usr/local/include, /opt, and /opt/local 
#   and evaluates the environment variable for googletest library and header files. 
#
# ADAPTED 
#   Yaser Afshar @ ya.afshar@gmail.com

AC_DEFUN([AX_GOOGLETEST], [
        AC_ARG_WITH([googletest], AS_HELP_STRING([--with-googletest@<:@=DIR@:>@], [use GOOGLETEST framework (default is no)]), 
                [ 
                        if test x$withval = xno ; then
                                AC_MSG_WARN([You can not test the library without GOOGLETEST framework !!!])
                        elif test x$withval = xyes ; then
                                want_googletest="yes"
                                ac_googletest_path=""
                        elif test x$withval != x ; then
                                want_googletest="yes"
                                ac_googletest_path="$withval"
                        else 
                                AC_MSG_WARN([You can not test the library without GOOGLETEST framework !!!])
                        fi
                ], [want_googletest="no"
                    ax_googletest_ok="no" ]
        )
   
        if test x$want_googletest = xyes; then
                succeeded=no
                
                dnl first we check the system location for googletest libraries
                if test x$ac_googletest_path != x; then
                        for ac_googletest_path_tmp in $ac_googletest_path $ac_googletest_path/include ; do 
                                if test -d "$ac_googletest_path_tmp/gtest" && test -r "$ac_googletest_path_tmp/gtest" ; then
                                        if test -f "$ac_googletest_path_tmp/gtest/gtest.h"  && test -r "$ac_googletest_path_tmp/gtest/gtest.h" ; then
                                                googletest_CPPFLAGS=" -I$ac_googletest_path_tmp"
                                                break;
                                        fi
                                fi
                        done
                else
                        for ac_googletest_path_tmp in /usr /usr/local /use/local/include /opt /opt/local ; do
                                if test -f "$ac_googletest_path_tmp/gtest.h" && test -r "$ac_googletest_path_tmp/gtest.h"; then
                                        googletest_CPPFLAGS=" -I$ac_googletest_path_tmp"
                                        break;
                                fi
                        done
                fi

                CPPFLAGS_SAVED="$CPPFLAGS"
                LDFLAGS_SAVED="$LDFLAGS"

                CPPFLAGS+="$googletest_CPPFLAGS"

                googletest_LDFLAGS=""
                if test x$ac_googletest_path != x; then
                        if test -d "$ac_googletest_path/lib" && test -r "$ac_googletest_path/lib" ; then
                                googletest_LDFLAGS=" -L$ac_googletest_path/lib"   
                        fi
                else
                        for ac_googletest_path_tmp in /usr/lib /usr/lib64 /use/local/lib /use/local/lib64 /opt /opt/lib ; do
                                if test -f "$ac_googletest_path_tmp/libgtest.so" && test -r "$ac_googletest_path_tmp/libgtest.so"; then
                                        googletest_LDFLAGS=" -L$ac_googletest_path_tmp" 
                                        break;
                                fi
                                if test -f "$ac_googletest_path_tmp/libgtest.a" && test -r "$ac_googletest_path_tmp/libgtest.a"; then
                                        googletest_LDFLAGS=" -L$ac_googletest_path_tmp" 
                                        break;
                                fi      
                        done
                fi

                LDFLAGS+="$googletest_LDFLAGS -lgtest"

                AC_LANG_PUSH(C++)
                
                AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                        @%:@include "gtest/gtest.h"
                ]], [[]]
                )], [AC_MSG_RESULT(yes)
                succeeded=yes
                found_system=yes
                ],[])

                LIBS_SAVED="$LIBS"
                
                AC_CHECK_LIB( [pthread], [main], 
                        [], [
                        succeeded=no
                        AC_MSG_ERROR([ Unable to continue! pthread devel library is missing! pthread is required for this program!])
                        ]
                )    
                
                AC_CHECK_LIB( [gtest], [main], 
                        [], [
                        succeeded=no
                        AC_MSG_ERROR([ Unable to continue! located googletest library does not work!])
                        ]
                )
                
                AC_LANG_POP([C++])

                if test x"$succeeded" == xyes ; then
                        GTEST_CPPFLAGS="$CPPFLAGS"
                        GTEST_CXXFLAGS="$CXXFLAGS"
                        GTEST_LDFLAGS="$LDFLAGS"                
                        GTEST_LIBS="$LIBS"
                                                                 
                        AC_SUBST(GTEST_CPPFLAGS)
                        AC_SUBST(GTEST_CXXFLAGS)
                        AC_SUBST(GTEST_LDFLAGS)
                        AC_SUBST(GTEST_LIBS)
                        
                        ax_googletest_ok="yes"
                fi
                
                CPPFLAGS="$CPPFLAGS_SAVED"
                LDFLAGS="$LDFLAGS_SAVED"                
                LIBS="$LIBS_SAVED" 
                AC_SUBST(LIBS)
        fi
])
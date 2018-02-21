#
# SYNOPSIS
#
#   AX_FLANN([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   Test for the FLANN libraries
#
#   If no path to the installed FLANN library is given the macro searchs
#   under /usr, /usr/local, /usr/local/include, /opt, and /opt/local 
#   and evaluates the environment variable for FLANN library and header files. 
#
# ADAPTED 
#   Yaser Afshar @ ya.afshar@gmail.com

AC_DEFUN([AX_FLANN], [
        AC_ARG_WITH([flann], 
                AS_HELP_STRING([--with-flann@<:@=DIR@:>@], 
                               [use FLANN library (default is yes) - it is possible to specify the root directory for FLANN (optional)]),            
                [ 
                        if test x$withval = xno ; then
                                AC_MSG_ERROR([ Unable to continue without the FLANN library !])
                        elif test x$withval = xyes ; then
                                want_flann="yes"
                                ac_flann_path=""
                        else
                                want_flann="yes"
                                ac_flann_path="$withval"
                        fi
                ], [want_flann="yes"]
        )
   
        if test x$want_flann = xyes; then
                succeeded=no
                
                dnl first we check the system location for flann libraries
                if test x$ac_flann_path != x; then
                        for ac_flann_path_tmp in $ac_flann_path $ac_flann_path/include ; do 
                                if test -d "$ac_flann_path_tmp/flann" && test -r "$ac_flann_path_tmp/flann" ; then
                                        if test -f "$ac_flann_path_tmp/flann/flann.h"  && test -r "$ac_flann_path_tmp/flann/flann.h" ; then
                                                flann_CPPFLAGS=" -I$ac_flann_path_tmp"
                                                break;
                                        fi
                                fi
                        done
                else
                        for ac_flann_path_tmp in /usr /usr/local /use/local/include /opt /opt/local ; do
                                if test -d "$ac_flann_path_tmp/flann" && test -r "$ac_flann_path_tmp/flann" ; then
                                        if test -f "$ac_flann_path_tmp/flann/flann.h" && test -r "$ac_flann_path_tmp/flann/flann.h"; then
                                                flann_CPPFLAGS=" -I$ac_flann_path_tmp/flann"
                                                break;
                                        fi
                                fi
                        done
                fi

                CPPFLAGS_SAVED="$CPPFLAGS"
                LDFLAGS_SAVED="$LDFLAGS"

                CPPFLAGS+="$flann_CPPFLAGS"

                flann_LDFLAGS=""
                if test x$ac_flann_path != x; then
                        if test -d "$ac_flann_path/lib" && test -r "$ac_flann_path/lib" ; then
                                flann_LDFLAGS=" -L$ac_flann_path/lib"   
                        fi
                else
                        for ac_flann_path_tmp in /usr/lib /usr/lib64 /use/local/lib /use/local/lib64 /opt /opt/lib ; do
                                if test -f "$ac_flann_path_tmp/libflann.so" && test -r "$ac_flann_path_tmp/libflann.so"; then
                                        flann_LDFLAGS=" -L$ac_flann_path_tmp" 
                                        break;
                                fi
                                if test -f "$ac_flann_path_tmp/libflann.a" && test -r "$ac_flann_path_tmp/libflann.a"; then
                                        flann_LDFLAGS=" -L$ac_flann_path_tmp" 
                                        break;
                                fi      
                        done
                fi

                LDFLAGS+="$flann_LDFLAGS -lflann"

                AC_LANG_PUSH(C++)
                AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                        @%:@include <flann/flann.h>
                ]], [[]]
                )], [AC_MSG_RESULT(yes)
                succeeded=yes
                found_system=yes
                ],[])
                
                AC_CHECK_LIB( flann, flann_build_index_double, 
                        [], [
                        succeeded=no
                        AC_MSG_ERROR([ Unable to continue without the FLANN library !])
                        ]
                )
                
                AC_LANG_POP([C++])

                if test "x$succeeded" == "xyes" ; then
                        AC_SUBST(CPPFLAGS)
                        AC_SUBST(LDFLAGS)
                        ax_flann_ok="yes"
                else
                        CPPFLAGS="$CPPFLAGS_SAVED"
                        LDFLAGS="$LDFLAGS_SAVED"
                fi
        fi
])
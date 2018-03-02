#
# SYNOPSIS
#
#   AX_TORC([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   Test for the TORC libraries
#
#   If no path to the installed TORC library is given the macro searchs
#   under /usr, /usr/local, /usr/local/include, /opt, and /opt/local 
#   and evaluates the environment variable for TORC library and header files. 
#
# ADAPTED 
#   Yaser Afshar @ ya.afshar@gmail.com

AU_ALIAS([ACX_TORC], [AX_TORC])
AC_DEFUN([AX_TORC], [
        AC_MSG_NOTICE()

        AC_ARG_WITH([torc], 
                AS_HELP_STRING([--with-torc@<:@=DIR@:>@], 
                               [use TORC library (default is yes) - it is possible to specify the PATH for TORC (optional)]),            
                [ 
                        if test x$withval = xno ; then
                                AC_MSG_ERROR([ Unable to continue without the TORC library !])
                        elif test x$withval = xyes ; then
                                want_torc="yes"
                                ac_torc_path=""
                        else
                                want_torc="yes"
                                ac_torc_path="$withval"
                        fi
                ], [want_torc="yes"]
        )
        
        if test x$want_torc = xyes; then              
                succeeded=no
                
                dnl first we check the system location for TORC libraries
                if test x$ac_torc_path != x; then
                        for ac_torc_path_tmp in $ac_torc_path $ac_torc_path/include $ac_torc_path/torc $ac_torc_path/torc/include ; do 
                                if test -f "$ac_torc_path_tmp/torc.h"  && test -r "$ac_torc_path_tmp/torc.h" ; then
                                        TORC_CFLAGS=" -I$ac_torc_path_tmp"" $PTHREAD_CFLAGS "
                                        break;
                                fi
                        done
                else                   
                        for ac_torc_path_tmp in /usr /usr/local /usr/include /use/local/include /opt /opt/local ; do
                                if test -d "$ac_torc_path_tmp/torc" && test -r "$ac_torc_path_tmp/torc" ; then
                                        if test -f "$ac_torc_path_tmp/torc/torc.h" && test -r "$ac_torc_path_tmp/torc/torc.h"; then
                                                TORC_CFLAGS=" -I$ac_torc_path_tmp/torc"" $PTHREAD_CFLAGS "
                                                break;
                                        fi
                                else
                                        if test -f "$ac_torc_path_tmp/torc.h" && test -r "$ac_torc_path_tmp/torc.h"; then
                                                TORC_CFLAGS=" -I$ac_torc_path_tmp"" $PTHREAD_CFLAGS "
                                                break;
                                        fi       
                                fi
                        done
                fi

                CFLAGS_SAVED="$CFLAGS"
                LDFLAGS_SAVED="$LDFLAGS"

                CFLAGS+="$TORC_CFLAGS"

                TORC_LDFLAGS=""
                if test x$ac_torc_path != x; then
                        if test -d "$ac_torc_path/lib" && test -r "$ac_torc_path/lib" ; then
                                TORC_LDFLAGS=" -L$ac_torc_path/lib"   
                        else
                                 if test -d "$ac_torc_path/torc/lib" && test -r "$ac_torc_path/torc/lib" ; then
                                        TORC_LDFLAGS=" -L$ac_torc_path/torc/lib"
                                 fi
                        fi
                else
                        for ac_torc_path_tmp in /usr/lib /usr/lib64 /use/local/lib /use/local/lib64 /opt /opt/lib ; do
                                if test -f "$ac_torc_path_tmp/libtorc.a" && test -r "$ac_torc_path_tmp/libtorc.a"; then
                                        TORC_LDFLAGS=" -L$ac_torc_path_tmp" 
                                        break;
                                fi      
                        done
                fi


                LDFLAGS+="$TORC_LDFLAGS"' -ltorc '" $PTHREAD_LIBS "         
                
                save_CC="$CC"
                if test x$ax_mpi_ok = xyes; then 
                        CC="$MPICC"
                fi

                AC_LANG_PUSH(C)

                AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
                        @%:@include <torc.h>
                ]], [[]]
                )], [AC_MSG_RESULT(yes)
                succeeded=yes
                found_system=yes
                ],[])
                
                AC_CHECK_LIB( torc, torc_init, 
                        [], [
                        succeeded=no
                        AC_MSG_ERROR([ Unable to continue without the TORC library !])
                        ]
                )
                
                AC_LANG_POP([C])

                CC="$save_CC"

                if test "x$succeeded" == "xyes" ; then
                        AC_SUBST(CFLAGS)
                        AC_SUBST(LDFLAGS)
                        ax_torc_ok="yes"
                else
                        CFLAGS="$CFLAGS_SAVED"
                        LDFLAGS="$LDFLAGS_SAVED"
                fi
        fi
        
        AC_MSG_RESULT()
])
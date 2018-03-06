#
# SYNOPSIS
#
#   AX_DOXYGEN([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   Test to see if the DOXYGEN has been installed.
#
# ADAPTED 
#  Yaser Afshar @ ya.afshar@gmail.com
#  Dept of Aerospace Engineering | University of Michigan

AU_ALIAS([ACX_DOXYGEN], [AX_DOXYGEN])
AC_DEFUN([AX_DOXYGEN], [
    AC_MSG_NOTICE(DOXYGEN)

    AC_ARG_ENABLE([doxygen], 
        AS_HELP_STRING([--enable-doxygen@<:@=ARG@:>@], 
            [Enable doxygen (default is no) (optional)]), 
        [ 
            if test x"$enableval" = xno ; then
                AC_MSG_WARN([You can not generate software reference documentation without DOXYGEN !!!])
            elif test x"$enableval" = xyes ; then
                want_doxygen="yes"
            elif test x"$enableval" != x ; then
                want_doxygen="yes"
            else 
                AC_MSG_WARN([You can not generate software reference documentation without DOXYGEN !!!])
            fi
        ], 
        [want_doxygen="no"]
    )
    
    ax_doxygen_ok="no"
    
    if test x"$want_doxygen" = xyes; then
        AC_CHECK_PROGS([DOXYGEN], [doxygen]) 
    
        AC_CHECK_PROGS([DOT], [dot])
        #if test -z "$DOT"; then
            #AC_MSG_ERROR([Doxygen needs dot, please install dot first])
        #fi
    
        AC_CHECK_PROGS([PDFLATEX], [pdflatex])
        if test -z "$PDFLATEX"; then
            AC_MSG_ERROR([Doxygen needs pdflatex program, it is part of TeX http://www.tug.org/texlive/acquire-netinstall.html])
        fi

        if test -z "$DOXYGEN"; then 
            AC_MSG_WARN([Doxygen not found - continuing without Doxygen support !])
            :
        else
            ax_doxygen_ok="yes"
            AC_DEFINE(HAVE_DOXYGEN, 1, [Define if you have DOXYGEN.])
            :
        fi
    fi
    
    AM_CONDITIONAL([HAVE_DOXYGEN], [test x"$ax_doxygen_ok" = xyes])
    AM_COND_IF([HAVE_DOXYGEN], [AC_CONFIG_FILES([docs/Doxyfile])], [AC_MSG_WARN([You can not generate software reference documentation without DOXYGEN !!!])])

    AC_MSG_RESULT()
])
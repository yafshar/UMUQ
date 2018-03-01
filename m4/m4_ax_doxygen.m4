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
#   Yaser Afshar @ ya.afshar@gmail.com

AU_ALIAS([ACX_DOXYGEN], [AX_DOXYGEN])
AC_DEFUN([AX_DOXYGEN], [
    AC_CHECK_PROGS([DOXYGEN], [doxygen]) 
    
    AC_CHECK_PROGS([DOT], [dot])
    if test -z "$DOT"; then
        AC_MSG_ERROR([Doxygen needs dot, please install dot first])
    fi
    
    AC_CHECK_PROGS([PDFLATEX], [pdflatex])
    if test -z "$PDFLATEX"; then
        AC_MSG_ERROR([Doxygen needs pdflatex program, it is part of TeX http://www.tug.org/texlive/acquire-netinstall.html])
    fi

    AM_CONDITIONAL([HAVE_DOXYGEN], [test -n "$DOXYGEN"])
    AM_COND_IF([HAVE_DOXYGEN], [AC_CONFIG_FILES([docs/Doxyfile])])

    if test -z "$DOXYGEN"; then 
        AC_MSG_WARN([Doxygen not found - continuing without Doxygen support !])
        :
    else
        AC_DEFINE(HAVE_DOXYGEN, 1, [Define if you have DOXYGEN.])
        :
    fi   
])
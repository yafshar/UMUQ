#
# SYNOPSIS
#
#   AX_EIGEN([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   Test for the EIGEN libraries
#
#   If no path to the installed eigen library is given the macro searchs
#   under /usr, /usr/local, /usr/local/include, /opt, and /opt/local 
#   and look for header files.
#
# ADAPTED 
#  Yaser Afshar @ ya.afshar@gmail.com
#  Dept of Aerospace Engineering | University of Michigan

AU_ALIAS([ACX_EIGEN], [AX_EIGEN])
AC_DEFUN([AX_EIGEN], [
	AC_ARG_WITH([eigen], 
		AS_HELP_STRING([--with-eigen@<:@=DIR@:>@], 
			[use EIGEN library (default is yes) - it is possible to specify the root directory for EIGEN (optional)]
		), [ 
			if test x"$withval" = xno ; then
				AC_MSG_ERROR([ Unable to continue without the EIGEN library !])
			elif test x"$withval" = xyes ; then
				ac_eigen_path=""
			elif test x"$withval" != x ; then
				ac_eigen_path="$withval"
            else
				ac_eigen_path=""
			fi
		], [
			ac_eigen_path=""
        ]
    )
  
  	dnl if the user does not provide the DIR root directory for EIGEN, we search the default PATH
	AS_IF([test x"$ac_eigen_path" = x], [ 
		AC_CHECK_HEADERS([Eigen/Dense], [ax_eigen_ok="yes"], [ax_eigen_ok="no"])
	], [
		ax_eigen_ok="no"
	]) 
        
    AS_IF([test x"$ax_eigen_ok" = xno], [  
    	AC_MSG_NOTICE(EIGEN)
                
        succeeded=no
                
        dnl first we check the system location for eigen libraries
		EIGEN_CPPFLAGS=""
        if test x"$ac_eigen_path" != x; then
        	for ac_eigen_path_tmp in $ac_eigen_path $ac_eigen_path/include $ac_eigen_path/include/eigen3 ; do
                if test -d "$ac_eigen_path_tmp/Eigen" && test -r "$ac_eigen_path_tmp/Eigen" ; then
                    if test -f "$ac_eigen_path_tmp/Eigen/Dense"  && test -r "$ac_eigen_path_tmp/Eigen/Dense" ; then
                        EIGEN_CPPFLAGS="-I$ac_eigen_path_tmp"
                        break;
                    fi
                fi
            done
    	else
            for ac_eigen_path_tmp in /usr /usr/inlude /usr/local /use/local/include /opt /opt/local ; do
                if test -d "$ac_eigen_path_tmp/Eigen" && test -r "$ac_eigen_path_tmp/Eigen"; then
                    if test -f "$ac_eigen_path_tmp/Eigen/Dense"  && test -r "$ac_eigen_path_tmp/Eigen/Dense"; then
                        EIGEN_CPPFLAGS="-I$ac_eigen_path_tmp"
                        break;
                    fi
                fi
                if test -d "$ac_eigen_path_tmp/eigen" && test -r "$ac_eigen_path_tmp/eigen" ; then
                    if test -d "$ac_eigen_path_tmp/eigen/Eigen" && test -r "$ac_eigen_path_tmp/eigen/Eigen"; then
                        if test -f "$ac_eigen_path_tmp/eigen/Eigen/Dense"  && test -r "$ac_eigen_path_tmp/eigen/Eigen/Dense"; then
                        	EIGEN_CPPFLAGS="-I$ac_eigen_path_tmp/eigen"
                            break;
                        fi
                    fi
                fi
                if test -d "$ac_eigen_path_tmp/eigen3" && test -r "$ac_eigen_path_tmp/eigen3"; then
                    if test -d "$ac_eigen_path_tmp/eigen3/Eigen" && test -r "$ac_eigen_path_tmp/eigen3/Eigen"; then
                        if test -f "$ac_eigen_path_tmp/eigen3/Eigen/Dense"  && test -r "$ac_eigen_path_tmp/eigen3/Eigen/Dense"; then
                        	EIGEN_CPPFLAGS="-I$ac_eigen_path_tmp/eigen3"
                        	break;
                        fi
                    fi
                fi
            done
        fi

        CPPFLAGS_SAVED="$CPPFLAGS"
        CPPFLAGS+=" $EIGEN_CPPFLAGS"
echo $CPPFLAGS
        AC_LANG_PUSH(C++)
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[ 
				@%:@include <Eigen/Dense>
            ]], [[]]
            )], [
                AC_MSG_RESULT(checking Eigen/Dense usability...  yes)
                AC_MSG_RESULT(checking Eigen/Dense presence... yes)
                AC_MSG_RESULT(checking for Eigen/Dense... yes)
                succeeded=yes
            ], [
                AC_MSG_RESULT(checking Eigen/Dense usability...  no)
                AC_MSG_RESULT(checking Eigen/Dense presence... no)
                AC_MSG_RESULT(checking for Eigen/Dense... no)
                AC_MSG_ERROR([ Unable to continue without the EIGEN header files !])
            ]
        )
        AC_LANG_POP([C++])

        if test x"$succeeded" == xyes ; then
            AC_SUBST(CPPFLAGS)
            ax_eigen_ok="yes"
            AC_DEFINE(HAVE_EIGEN, 1, [Define if you have EIGEN Library.])
            :
        else
            ax_eigen_ok="no"
            CPPFLAGS="$CPPFLAGS_SAVED"
            :
        fi
    ])

    AS_IF([test x"$ax_eigen_ok" = xno], [ AC_MSG_ERROR([ Unable to find the EIGEN library !])])
    AC_MSG_RESULT()
])
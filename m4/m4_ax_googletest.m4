#
# SYNOPSIS
#
#	AX_GOOGLETEST([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#	Test for the GOOGLETEST framework
#
#	If no path to the installed googletest framework is given the macro searchs
#	under /usr, /usr/local, /usr/local/include, /opt, and /opt/local 
#	and evaluates the environment variable for googletest library and header files. 
#
# ADAPTED 
#	Yaser Afshar @ ya.afshar@gmail.com
#	Dept of Aerospace Engineering | University of Michigan

AU_ALIAS([ACX_GOOGLETEST], [AX_GOOGLETEST])
AC_DEFUN([AX_GOOGLETEST], [
	AC_ARG_WITH([googletest], 
		AS_HELP_STRING([--with-googletest@<:@=DIR@:>@], 
			[use GOOGLETEST framework (default is no) - it is possible to specify the root directory for GOOGLETEST (optional)]
		), [
			if test x"$withval" = xno ; then
				AC_MSG_WARN([You can not test the library without GOOGLETEST framework !!!])
				ac_googletest_path=no
			elif test x"$withval" = xyes ; then
				ac_googletest_path=""
			elif test x"$withval" != x ; then
				ac_googletest_path="$withval"
			else 
				ac_googletest_path=""
			fi
		], [
			ac_googletest_path=no
		]
	)

	AS_IF([test x"$ac_googletest_path" = xno], [], 
		[
			CPPFLAGS_SAVED="$CPPFLAGS"

			dnl if the user does not provide the DIR root directory for EIGEN, we search the default PATH
			AS_IF([test x"$ac_googletest_path" = x], 
				[ 
					AC_CHECK_HEADERS([gtest/gtest.h], 
						[
							ax_googletest_ok=yes
							succeeded=yes
						], [
							ax_googletest_ok=no
							succeeded=no
						]
					)
				], [
					ax_googletest_ok=no
					succeeded=no
				]
			)
			
			AS_IF([test x"$ax_googletest_ok" = xno], 
				[
					AC_MSG_NOTICE(GOOGLETEST)

					dnl first we check the system location for googletest libraries
					if test x"$ac_googletest_path" != x; then
						for ac_googletest_path_tmp in $ac_googletest_path $ac_googletest_path/include ; do 
							if test -d "$ac_googletest_path_tmp/gtest" && test -r "$ac_googletest_path_tmp/gtest" ; then
								if test -f "$ac_googletest_path_tmp/gtest/gtest.h"  && test -r "$ac_googletest_path_tmp/gtest/gtest.h" ; then
									googletest_CPPFLAGS=" -I$ac_googletest_path_tmp"
									break;
								fi
							fi
						done
					else
						for ac_googletest_path_tmp in /usr /usr/inlude /usr/local /use/local/include /opt /opt/local ; do
							if test -f "$ac_googletest_path_tmp/gtest.h" && test -r "$ac_googletest_path_tmp/gtest.h"; then
								googletest_CPPFLAGS=" -I$ac_googletest_path_tmp"
								break;
							fi
						done
					fi

					CPPFLAGS+="$googletest_CPPFLAGS"

					AC_LANG_PUSH(C++)
					AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
							@%:@include "gtest/gtest.h"
						]], [[]]
						)], [
							AC_MSG_RESULT(checking gtest/gtest.h usability...  yes)
							AC_MSG_RESULT(checking gtest/gtest.h presence... yes)
							AC_MSG_RESULT(checking for gtest/gtest.h... yes)
							succeeded=yes
						], [
							AC_MSG_RESULT(checking gtest/gtest.husability...  no)
							AC_MSG_RESULT(checking gtest/gtest.h presence... no)
							AC_MSG_RESULT(checking for gtest/gtest.h... no)
							AC_MSG_ERROR([ Unable to continue without the GOOGLETEST header files !])
						]
					)
					AC_LANG_POP([C++])
				]
			)

			if test x"$succeeded" == xyes ; then
				GTEST_CPPFLAGS="$CPPFLAGS"
				GTEST_CXXFLAGS="$CXXFLAGS"
				GTEST_LDFLAGS="$LDFLAGS"
				GTEST_LIBS="$LIBS"

				AC_SUBST(GTEST_CPPFLAGS)
				AC_SUBST(GTEST_CXXFLAGS)
				AC_SUBST(GTEST_LDFLAGS)
				AC_SUBST(GTEST_LIBS)

				ax_googletest_ok="yes"$
			fi

			CPPFLAGS="$CPPFLAGS_SAVED"
		]
	)

	AM_CONDITIONAL([HAVE_GOOGLETEST], [test x"$ax_googletest_ok" == xyes])
	AM_COND_IF([HAVE_GOOGLETEST],
		[], [
			AC_MSG_RESULT(Unable to locate GOOGLETEST !) 
			AC_MSG_RESULT(You can not test the library without GOOGLETEST framework !!!)
		]
	)

	AC_MSG_RESULT()
])

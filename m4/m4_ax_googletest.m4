#
# SYNOPSIS
#
#	AX_GOOGLETEST([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#	Test for the GOOGLETEST framework
#
#	If no path to the installed googletest framework is given the macro uses
#	the external installation and evaluates the environment variable for 
#	googletest library and header files. 
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
			LDFLAGS_SAVED="$LDFLAGS"
			CPPFLAGS_SAVED="$CPPFLAGS"
			LIBS_SAVED="$LIBS"

			dnl if the user does not provide the DIR root directory for googletest, we search the default PATH
			AS_IF([test x"$ac_googletest_path" = x],
				[
					AC_CHECK_HEADERS([gtest/gtest.h], [ax_googletest_ok=yes], [ax_googletest_ok=no])

					AS_IF([test x"$ax_googletest_ok" = xyes],
						[
							AC_LANG_PUSH(C++)
							LDFLAGS+=' -lgtest'
							AC_CHECK_LIB([gtest], [main], 
								[
									succeeded=yes
								], [
									ax_googletest_ok=no
									succeeded=no
									LDFLAGS="$LDFLAGS_SAVED"
								]
							)
							AC_LANG_POP([C++])
						]
					)
				],
				[
					ax_googletest_ok=no
					succeeded=no
				]
			)

			AS_IF([test x"$ax_googletest_ok" = xno], 
				[
					AC_MSG_NOTICE(GOOGLETEST)

					googletest_LDFLAGS=
					googletest_CPPFLAGS=
					googletest_PATH=

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
						for ac_googletest_path_tmp in external ; do
							if test -d "$ac_googletest_path_tmp/googletest" && test -r "$ac_googletest_path_tmp/googletest" ; then
								googletest_PATH=`pwd`
								googletest_PATH+='/'"$ac_googletest_path_tmp"'/googletest'
								if test -f "$googletest_PATH/googletest/include/gtest/gtest.h" && test -r "$googletest_PATH/googletest/include/gtest/gtest.h"; then
									googletest_CPPFLAGS=" -I$googletest_PATH"'/googletest/include'
									break;
								fi
							fi
						done
					fi

					CPPFLAGS+="$googletest_CPPFLAGS"

					if test x"$ac_googletest_path" != x; then
						if test -d "$ac_googletest_path/lib" && test -r "$ac_googletest_path/lib" ; then
							googletest_LDFLAGS=" -L$ac_googletest_path/lib"   
						fi
					else
						if test x"$googletest_PATH" != x ; then
							if test -f "$googletest_PATH/googlemock/gtest/libgtest.a" && test -r "$googletest_PATH/googlemock/gtest/libgtest.a"; then
								googletest_LDFLAGS=" -L$googletest_PATH"'/googlemock/gtest'
							fi
							if test -f "$googletest_PATH/googlemock/gtest/libgtest.so" && test -r "$googletest_PATH/googlemock/gtest/libgtest.so"; then
								googletest_LDFLAGS=" -L$googletest_PATH"'/googlemock/gtest'
							fi
							if test -f "$googletest_PATH/googlemock/gtest/libgtest.dyld" && test -r "$googletest_PATH/googlemock/gtest/libgtest.dyld"; then
								googletest_LDFLAGS=" -L$googletest_PATH"'/googlemock/gtest'
							fi
							if test x"$googletest_LDFLAGS" = x ; then
								(cd "$googletest_PATH" && cmake CMakeLists.txt && make)
								googletest_LDFLAGS=" -L$googletest_PATH"'/googlemock/gtest'
							fi
						fi
					fi

					LDFLAGS+="$googletest_LDFLAGS"' -lgtest'

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

					AC_CHECK_LIB([gtest], [main], 
						[], [
							succeeded=no
							AC_MSG_ERROR([ Unable to continue! located googletest library does not work!])
						]
					)
					AC_LANG_POP([C++])
				]
			)
		]

		if test x"$succeeded" == xyes ; then
			GTEST_CPPFLAGS="$CPPFLAGS"
			GTEST_CXXFLAGS="$CXXFLAGS"
			GTEST_LDFLAGS="$LDFLAGS"
			GTEST_LIBS="$LIBS"

			AC_SUBST(GTEST_CPPFLAGS)
			AC_SUBST(GTEST_CXXFLAGS)
			AC_SUBST(GTEST_LDFLAGS)
			AC_SUBST(GTEST_LIBS)

			ax_googletest_ok=yes
		fi

		CPPFLAGS="$CPPFLAGS_SAVED"
		LDFLAGS="$LDFLAGS_SAVED"
		LIBS="$LIBS_SAVED"

		AC_SUBST(LIBS)
	)


	AM_CONDITIONAL([HAVE_GOOGLETEST], [test x"$ax_googletest_ok" = xyes])
	AM_COND_IF([HAVE_GOOGLETEST],
		[], [
			AC_MSG_RESULT(Unable to locate GOOGLETEST !)
			AC_MSG_RESULT(You can not test the library without GOOGLETEST framework !!!)
		]
	)

	AC_MSG_RESULT()
])

#
# SYNOPSIS
#
#	AX_FLANN([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#	Test for the FLANN libraries
#
#	If no path to the installed FLANN library is given the macro uses
#	external folder and creates the FLANN library and header files. 
#
# AUTHOR
#	Yaser Afshar @ ya.afshar@gmail.com
#	Dept of Aerospace Engineering | University of Michigan

AU_ALIAS([ACX_FLANN], [AX_FLANN])
AC_DEFUN([AX_FLANN], [
	AC_ARG_WITH([flann], 
		AS_HELP_STRING([--with-flann@<:@=DIR@:>@], 
				[use FLANN library (default is yes) - it is possible to specify the root directory for FLANN (optional)]
		), [ 
			if test x"$withval" = xno ; then
				AC_MSG_ERROR([ Unable to continue without the FLANN library !])
			elif test x"$withval" = xyes ; then
				ac_flann_path=""
			elif test x"$withval" != x ; then
				ac_flann_path="$withval"
			else
				ac_flann_path=""
			fi
		], [
			ac_flann_path=""
		]
	)

	FLANN_LD_LIBRARY_PATH=

	dnl if the user does not provide the DIR root directory for FLANN, we search the default PATH
	AS_IF([test x"$ac_flann_path" = x], [ 
		AC_CHECK_HEADERS([flann/flann.h], [ax_flann_ok="yes"], [ax_flann_ok="no"])
		AS_IF([test x"$ax_flann_ok" = xyes], [
				LDFLAGS_SAVED="$LDFLAGS"
				LDFLAGS+=' -lflann'
				AC_LANG_PUSH(C++)
				AC_CHECK_LIB(flann, flann_build_index_double, [], 
					[
						ax_flann_ok="no"
						LDFLAGS="$LDFLAGS_SAVED"
					]
				)
				AC_LANG_POP([C++])
				AC_SUBST(LDFLAGS)
			]
		)
		], [
			ax_flann_ok="no"
		]
	)

	AS_IF([test x"$ax_flann_ok" = xno], [
		AC_MSG_NOTICE(FLANN)

		succeeded=no
		
		flann_LDFLAGS=
		flann_CPPFLAGS=
		flann_PATH=
		
		if test x"$ac_flann_path" != x; then
			for ac_flann_path_tmp in $ac_flann_path $ac_flann_path/include ; do 
				if test -d "$ac_flann_path_tmp/flann" && test -r "$ac_flann_path_tmp/flann" ; then
					if test -f "$ac_flann_path_tmp/flann/flann.h"  && test -r "$ac_flann_path_tmp/flann/flann.h" ; then
						flann_CPPFLAGS="-I$ac_flann_path_tmp"
						break;
					fi
				fi
			done
		else
			for ac_flann_path_tmp in external ; do
				if !( test -d "$ac_flann_path_tmp/flann/src" && test -r "$ac_flann_path_tmp/flann/src") ; then
					git submodule update --init external/flann
				fi
				if test -d "$ac_flann_path_tmp/flann/src" && test -r "$ac_flann_path_tmp/flann/src" ; then
					flann_PATH=`pwd`
					flann_PATH+='/'"$ac_flann_path_tmp"'/flann'
					if test -f "$flann_PATH/src/cpp/flann/flann.h" && test -r "$flann_PATH/src/cpp/flann/flann.h"; then 
						flann_CPPFLAGS="-I$flann_PATH"'/src/cpp'
						break;
					fi
				fi
			done
		fi

		CPPFLAGS_SAVED="$CPPFLAGS"
		LDFLAGS_SAVED="$LDFLAGS"

		CPPFLAGS+=" $flann_CPPFLAGS"

		if test x"$ac_flann_path" != x; then
			if test -d "$ac_flann_path/lib" && test -r "$ac_flann_path/lib" ; then
				flann_LDFLAGS=" -L$ac_flann_path/lib"
				FLANN_LD_LIBRARY_PATH="$ac_flann_path/lib"
			fi
		else
			if test x"$flann_PATH" != x ; then
				if test -f "$flann_PATH/build/lib/libflann.so" && test -r "$flann_PATH/build/lib/libflann.so"; then
					flann_LDFLAGS=" -L$flann_PATH"'/build/lib'
					FLANN_LD_LIBRARY_PATH="$flann_PATH"'/build/lib'
				fi
				if test -f "$flann_PATH/build/lib/libflann.a" && test -r "$flann_PATH/build/lib/libflann.a"; then
					flann_LDFLAGS=" -L$flann_PATH"'/build/lib'
					FLANN_LD_LIBRARY_PATH="$flann_PATH"'/build/lib'
				fi
				if test -f "$flann_PATH/build/lib/libflann.dylib" && test -r "$flann_PATH/build/lib/libflann.dylib"; then
					flann_LDFLAGS=" -L$flann_PATH"'/build/lib'
					FLANN_LD_LIBRARY_PATH="$flann_PATH"'/build/lib'
				fi
				if test x"$flann_LDFLAGS" = x ; then
					AC_LANG_PUSH([C++])
					(cd "$flann_PATH" && rm -fr build && mkdir -p build && cd build && export CC=$CC && export CXX=$CXX && cmake CC=$CC CXX=$CXX -DCMAKE_INSTALL_PREFIX="$flann_PATH" ../ && make -j 4)
					flann_LDFLAGS=" -L$flann_PATH"'/build/lib'
					FLANN_LD_LIBRARY_PATH="$flann_PATH"'/build/lib'
					AC_LANG_POP([C++])
				fi
			fi
		fi

		LDFLAGS+="$flann_LDFLAGS"' -lflann'

		AC_LANG_PUSH([C++])
		AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
				@%:@include <flann/flann.h>
			]], [[]]
			)], [
				AC_MSG_RESULT(checking flann/flann.h usability...  yes)
				AC_MSG_RESULT(checking flann/flann.h presence... yes)
				AC_MSG_RESULT(checking for flann/flann.h... yes)
				succeeded=yes
			], [
				AC_MSG_RESULT(checking flann/flann.h usability...  no)
				AC_MSG_RESULT(checking flann/flann.h presence... no)
				AC_MSG_RESULT(checking for flann/flann.h... no)
				AC_MSG_ERROR([ Unable to continue without the FLANN header files !])
			]
		)

		LIBS_SAVED="$LIBS"
		AC_CHECK_LIB(flann, flann_build_index_double, 
			[
				LDFLAGS+=' -Wl,-rpath,'"$FLANN_LD_LIBRARY_PATH"
			], [
				succeeded=no
				AC_MSG_ERROR([ Unable to continue without the FLANN library !])
			]
		)
		AS_IF([test x"$LIBS" = x"$LIBS_SAVED"], [LIBS='-lflann'" $LIBS"])
		AC_LANG_POP([C++])

		if test x"$succeeded" = xyes ; then
			AC_SUBST(CPPFLAGS)
			AC_SUBST(LDFLAGS)
			AC_SUBST(FLANN_LD_LIBRARY_PATH)
			ax_flann_ok="yes"
			:
		else
			ax_flann_ok="no"
			CPPFLAGS="$CPPFLAGS_SAVED"
			LDFLAGS="$LDFLAGS_SAVED"
			:
		fi
	])

	AS_IF([test x"$ax_flann_ok" = xno], [ AC_MSG_ERROR([ Unable to find the FLANN library !])])
	AM_CONDITIONAL([HAVE_FLANN], [test x"$ax_flann_ok" = xyes])
	AM_COND_IF([HAVE_FLANN], [
			AC_DEFINE(HAVE_FLANN, 1, [Define if you have FLANN Library.])
		]
	)

	AC_MSG_RESULT()
])

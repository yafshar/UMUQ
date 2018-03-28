#
# SYNOPSIS
#
#	AX_TORC([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#	Test for the TORC libraries
#
#	If no path to the installed TORC library is given the macro searchs
#	under /usr, /usr/local, /usr/local/include, /opt, and /opt/local 
#	and evaluates the environment variable for TORC library and header files. 
#
# AUTHOR 
#	Yaser Afshar @ ya.afshar@gmail.com
#	Dept of Aerospace Engineering | University of Michigan

AU_ALIAS([ACX_TORC], [AX_TORC])
AC_DEFUN([AX_TORC], [
	AC_ARG_WITH([torc], 
		AS_HELP_STRING([--with-torc@<:@=DIR@:>@], 
			[use TORC library (default is yes) - it is possible to specify the root directory for TORC library (optional)]),
		[ 
			if test x"$withval" = xno; then
				AC_MSG_ERROR([ Unable to continue without the TORC library !])
			elif test x"$withval" = xyes; then
				ac_torc_path=
			elif test x"$withval" != x; then
				ac_torc_path="$withval"
			else
				ac_torc_path=
			fi
		], [ac_torc_path=]
	)

	dnl if the user does not provide the DIR root directory for torc, we search the default PATH
	AS_IF([test x"$ac_torc_path" = x], [ 
		AC_CHECK_HEADERS([torc.h], [ax_torc_ok=yes], [ax_torc_ok=no])

		AS_IF([test x"$ax_torc_ok" = xyes], [
				LDFLAGS_SAVED="$LDFLAGS"
				LDFLAGS+=' -ltorc '" $PTHREAD_LIBS "
				AC_LANG_PUSH(C)
				AC_CHECK_LIB(torc, torc_init, [], 
					[
						ax_torc_ok=no
						LDFLAGS="$LDFLAGS_SAVED"
					]
				)
				AC_LANG_POP([C])
				AC_SUBST(LDFLAGS)
			])
		], [
			ax_torc_ok=no
		]
	)

	AS_IF([test x"$ax_torc_ok" = xno], [  
		AC_MSG_NOTICE(TORC)

		succeeded=no

		torc_CFLAGS=
		torc_LDFLAGS=
		torc_PATH=

		dnl first we check the system location for TORC libraries
		if test x"$ac_torc_path" != x; then
			for ac_torc_path_tmp in $ac_torc_path $ac_torc_path/include $ac_torc_path/torc $ac_torc_path/torc/include ; do 
				if test -d "$ac_torc_path_tmp" && test -r "$ac_torc_path_tmp" ; then
					if test -f "$ac_torc_path_tmp/torc.h" && test -r "$ac_torc_path_tmp/torc.h" ; then
						torc_CFLAGS="-I$ac_torc_path_tmp"" $PTHREAD_CFLAGS "
						break;
					fi
				fi
			done
		else                   
			for ac_torc_path_tmp in external ; do
				if !( test -d "$ac_torc_path_tmp/torc/src" && test -r "$ac_torc_path_tmp/torc/src") ; then
					git submodule update --init external/torc
				fi
				if test -d "$ac_torc_path_tmp/torc/src" && test -r "$ac_torc_path_tmp/torc/src" ; then
					torc_PATH=`pwd`
					torc_PATH+='/'"$ac_torc_path_tmp"'/torc'
					if test -f "$torc_PATH/include/torc.h" && test -r "$torc_PATH/include/torc.h"; then 
						torc_CFLAGS="-I$torc_PATH/include"" $PTHREAD_CFLAGS "
						break;
					fi
				fi	
			done
		fi

		CFLAGS_SAVED="$CFLAGS"
		LDFLAGS_SAVED="$LDFLAGS"

		CFLAGS+=" $torc_CFLAGS"

		if test x"$ac_torc_path" != x; then
			for ac_torc_path_tmp in $ac_torc_path $ac_torc_path/lib $ac_torc_path/torc $ac_torc_path/torc/lib ; do
				if test -d "$ac_torc_path_tmp" && test -r "$ac_torc_path_tmp" ; then
					if test -f "$ac_torc_path_tmp/libtorc.a" && test -r "$ac_torc_path_tmp/libtorc.a"; then
						torc_LDFLAGS="-L$ac_torc_path_tmp" 
						break;
					fi
					if test -f "$ac_torc_path_tmp/libtorc.dylib" && test -r "$ac_torc_path_tmp/libtorc.dylib"; then
						torc_LDFLAGS="-L$ac_torc_path_tmp" 
						break;
					fi
				fi
			done
		else
			if test x"$torc_PATH" != x ; then
				if test -f "$torc_PATH/src/libtorc.so" && test -r "$torc_PATH/src/libtorc.so"; then
					torc_LDFLAGS=" -L$torc_PATH"'/src' 
				fi
				if test -f "$torc_PATH/src/libtorc.a" && test -r "$torc_PATH/src/libtorc.a"; then
					torc_LDFLAGS=" -L$torc_PATH"'/src'
				fi
				if test -f "$torc_PATH/src/libtorc.dylib" && test -r "$torc_PATH/src/libtorc.dylib"; then
					torc_LDFLAGS=" -L$torc_PATH"'/src' 
				fi
				if test x"$torc_LDFLAGS" = x ; then
					save_CC="$CC"
					AS_IF([test x"$ax_mpi_ok" = xyes], [CC="$MPICC"])
					AC_LANG_PUSH([C])
					(cd "$torc_PATH" && ./configure && make)
					torc_LDFLAGS=" -L$torc_PATH"'/src'
					AC_LANG_POP([C])
					CC="$save_CC"
				fi
			fi
		fi

		LDFLAGS+=" $torc_LDFLAGS"' -ltorc '" $PTHREAD_LIBS "

		save_CC="$CC"
		AS_IF([test x"$ax_mpi_ok" = xyes], [CC="$MPICC"])

		AC_LANG_PUSH([C])
		AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
				@%:@include <torc.h>
			]], [[]]
			)], [
				AC_MSG_RESULT(checking torc.h usability...  yes)
				AC_MSG_RESULT(checking torc.h presence... yes)
				AC_MSG_RESULT(checking for torc.h... yes)
				succeeded=yes
			],[
				AC_MSG_RESULT(checking torc.h usability...  no)
				AC_MSG_RESULT(checking torc.h presence... no)
				AC_MSG_RESULT(checking for torc.h... no)
				AC_MSG_ERROR([ Unable to continue without the TORC header files !])
			]
		)

		AC_CHECK_LIB(torc, torc_init, 
			[], [
				succeeded=no
				AC_MSG_ERROR([ Unable to continue without the TORC library !])
			]
		)
		AC_LANG_POP([C])

		CC="$save_CC"

		if test x"$succeeded" = xyes ; then
			AC_SUBST(CFLAGS)
			AC_SUBST(LDFLAGS)
			ax_torc_ok=yes
		else
			CFLAGS="$CFLAGS_SAVED"
			LDFLAGS="$LDFLAGS_SAVED"
		fi

	])

	AS_IF([test x"$ax_torc_ok" = xno], [ AC_MSG_ERROR([ Unable to find the TORC library !])])
	AC_MSG_RESULT()
])

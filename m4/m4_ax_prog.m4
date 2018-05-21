#
# SYNOPSIS
#
#	AX_PROG([]])
#	AX_PROG_SUBSTITUTE([]])
#   AX_PROG_CPP_CORRECTION([])
#
# DESCRIPTION
#
#
# AUTHOR 
#	Yaser Afshar @ ya.afshar@gmail.com
#	Dept of Aerospace Engineering | University of Michigan

AC_DEFUN([AX_PROG], [
	AC_PROG_CXX()
	AC_PROG_CC()
	AC_PROG_FC()
	AC_SUBST(F77,[$FC])
	AC_PROG_CPP
	AC_PROG_RANLIB
	AC_PROG_MKDIR_P
	AC_PROG_MAKE_SET
])

AC_DEFUN([AX_PROG_SUBSTITUTE], [
	AC_SUBST(CXX,["$MPICXX"])
	AC_SUBST(CC,["$MPICC"])
	AC_SUBST(FC,["$MPIFC"])
	AC_SUBST(F77,[$FC])
	CPP=
	AX_PROG_CPP_CORRECTION
])

AC_DEFUN([AX_PROG_CPP_CORRECTION], [
	if test -z "$CPP"; then
		ac_ext=c
		ac_cpp='$CPP $CPPFLAGS'
		ac_compile='$CC -c $CFLAGS $CPPFLAGS conftest.$ac_ext >&5'
		ac_link='$CC -o conftest$ac_exeext $CFLAGS $CPPFLAGS $LDFLAGS conftest.$ac_ext $LIBS >&5'
		ac_compiler_gnu=$ac_cv_c_compiler_gnu

		ac_preproc_ok=false
		# Double quotes because CPP needs to be expanded
		for CPP in "$CC -E" "$CC -E -traditional-cpp" "/lib/cpp"
		do
			ac_preproc_ok=false
			for ac_c_preproc_warn_flag in '' yes
			do
				# Use a header file that comes with gcc, so configuring glibc
				# with a fresh cross-compiler works.
				# Prefer <limits.h> to <assert.h> if __STDC__ is defined, since
				# <limits.h> exists even on freestanding compilers.
				# On the NeXT, cc -E runs the code through the compiler's parser,
				# not just through cpp. "Syntax error" is here to catch this case.
				cat confdefs.h - <<_ACEOF >conftest.$ac_ext
/* end confdefs.h.  */
#ifdef __STDC__
#include <limits.h>
#else
#include <assert.h>
#endif
Syntax error
_ACEOF

				if ac_fn_c_try_cpp "$LINENO"; then :
				else
					# Broken: fails on valid input.
					continue
				fi
				rm -f conftest.err conftest.i conftest.$ac_ext

				# OK, works on sane cases.  Now check whether nonexistent headers
				# can be detected and how.
				cat confdefs.h - <<_ACEOF >conftest.$ac_ext
/* end confdefs.h.  */
#include <ac_nonexistent.h>
_ACEOF
				if ac_fn_c_try_cpp "$LINENO"; then :
					# Broken: success on invalid input.
					continue
				else
					# Passes both tests.
					ac_preproc_ok=:
					break
				fi
				rm -f conftest.err conftest.i conftest.$ac_ext
			done

			# Because of `break', _AC_PREPROC_IFELSE's cleaning code was skipped.
			rm -f conftest.i conftest.err conftest.$ac_ext
			if $ac_preproc_ok; then :
				break
			fi
		done
		if $ac_preproc_ok; then :
			ac_cv_prog_CPP=$CPP
		fi
	fi #if test -z "$CPP"; then

	AC_PROG_CPP
])

#
# SYNOPSIS
#
#   AX_PTHREAD([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
#
# DESCRIPTION
#
#   This macro figures out how to build C programs using POSIX threads. It
#   sets the PTHREAD_LIBS output variable to the threads library and linker
#   flags, and the PTHREAD_CFLAGS output variable to any special C compiler
#   flags that are needed. (The user can also force certain compiler
#   flags/libs to be tested by setting these environment variables.)
#
#   Also sets PTHREAD_CC to any special C compiler that is needed for
#   multi-threaded programs (defaults to the value of CC otherwise). (This is
#   necessary on AIX to use the special cc_r compiler alias.)
#   
#   NOTE: You are assumed to not only compile your program with these flags,
#   but also link it with them as well. e.g. you should link with $PTHREAD_CC
#   $CFLAGS $PTHREAD_CFLAGS $LDFLAGS ... $PTHREAD_LIBS $LIBS
#   
#   If you are only building threads programs, you may wish to use these
#   variables in your default LIBS, CFLAGS, and CC:
#   
#          LIBS="$PTHREAD_LIBS $LIBS"
#          CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
#          CC="$PTHREAD_CC"
#   
#   In addition, if the PTHREAD_CREATE_JOINABLE thread-attribute constant
#   has a nonstandard name, defines PTHREAD_CREATE_JOINABLE to that name
#   (e.g. PTHREAD_CREATE_UNDETACHED on AIX).
#   
#   ACTION-IF-FOUND is a list of shell commands to run if a threads library
#   is found, and ACTION-IF-NOT-FOUND is a list of commands to run it if it
#   is not found. If ACTION-IF-FOUND is not specified, the default action
#   will define HAVE_PTHREAD.
#   
#   Please let the authors know if this macro fails on any platform, or if
#   you have any other suggestions or comments. This macro was based on work
#   by SGJ on autoconf scripts for FFTW (www.fftw.org) (with help from M.
#   Frigo), as well as ac_pthread and hb_pthread macros posted by AFC to the
#   autoconf macro repository. We are also grateful for the helpful feedback
#   of numerous users.
#   
#   Version: 1.8 (last modified: 2003-05-21)
#   Author: Steven G. Johnson <stevenj@alum.mit.edu> and
#           Alejandro Forero Cuervo <bachue@bachue.com>
#   
#   from http://www.gnu.org/software/ac-archive/htmldoc/index.html
#   
#   License:
#   GNU General Public License
#   [http://www.gnu.org/software/ac-archive/htmldoc/COPYING.html]
#   with this special exception
#   [http://www.gnu.org/software/ac-archive/htmldoc/COPYING-Exception.html]. 
#
# ADAPTED 
#   Yaser Afshar @ ya.afshar@gmail.com

AU_ALIAS([ACX_PTHREAD], [AX_PTHREAD])
AC_DEFUN([AX_PTHREAD], [
        AC_REQUIRE([AC_CANONICAL_HOST])
        AC_LANG_PUSH(C)
        ax_pthread_ok=no

        # We used to check for pthread.h first, but this fails if pthread.h
        # requires special compiler flags (e.g. on True64 or Sequent).
        # It gets checked for in the link test anyway.

        # First of all, check if the user has set any of the PTHREAD_LIBS,
        # etcetera environment variables, and if threads linking works using
        # them:
        if test x"$PTHREAD_LIBS$PTHREAD_CFLAGS" != x; then
                save_CFLAGS="$CFLAGS"
                CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
                save_LIBS="$LIBS"
                LIBS="$PTHREAD_LIBS $LIBS"
                AC_MSG_CHECKING([for pthread_join in LIBS=$PTHREAD_LIBS with CFLAGS=$PTHREAD_CFLAGS])
                AC_TRY_LINK_FUNC(pthread_join, ax_pthread_ok=yes) 
                AC_MSG_RESULT($ax_pthread_ok)        
                if test x"$ax_pthread_ok" = xno; then
                        PTHREAD_LIBS=""
                        PTHREAD_CFLAGS=""
                fi
                LIBS="$save_LIBS"
                CFLAGS="$save_CFLAGS"
        fi

        # We must check for the threads library under a number of different
        # names; the ordering is very important because some systems
        # (e.g. DEC) have both -lpthread and -lpthreads, where one of the
        # libraries is broken (non-POSIX).

        # Create a list of thread flags to try.  Items starting with a "-" are
        # C compiler flags, and other items are library names, except for "none"
        # which indicates that we try without any flags at all.

        ax_pthread_flags="pthreads none -Kthread -kthread lthread -pthread -pthreads -mthreads pthread --thread-safe -mt"

        # The ordering *is* (sometimes) important.  Some notes on the
        # individual items follow:

        # pthreads: AIX (must check this before -lpthread)
        # none: in case threads are in libc; should be tried before -Kthread and
        #       other compiler flags to prevent continual compiler warnings
        # -Kthread: Sequent (threads in libc, but -Kthread needed for pthread.h)
        # -kthread: FreeBSD kernel threads (preferred to -pthread since SMP-able)
        # lthread: LinuxThreads port on FreeBSD (also preferred to -pthread)
        # -pthread: Linux/gcc (kernel threads), BSD/gcc (userland threads)
        # -pthreads: Solaris/gcc
        # -mthreads: Mingw32/gcc, Lynx/gcc
        # -mt: Sun Workshop C (may only link SunOS threads [-lthread], but it
        #      doesn't hurt to check since this sometimes defines pthreads too;
        #      also defines -D_REENTRANT)
        # pthread: Linux, etcetera
        # --thread-safe: KAI C++

        case "${host_cpu}-${host_os}" in
        *solaris*)
                # On Solaris (at least, for some versions), libc contains stubbed
                # (non-functional) versions of the pthreads routines, so link-based
                # tests will erroneously succeed.  (We need to link with -pthread or
                # -lpthread.)  (The stubs are missing pthread_cleanup_push, or rather
                # a function called by this macro, so we could check for that, but
                # who knows whether they'll stub that too in a future libc.)  So,
                # we'll just look for -pthreads and -lpthread first:

                ax_pthread_flags="-pthread -pthreads pthread -mt $ax_pthread_flags"
                ;;
        esac

        if test x"$ax_pthread_ok" = xno; then
                for flag in $ax_pthread_flags; do
                        case $flag in
                        none)
                                AC_MSG_CHECKING([whether pthreads work without any flags])
                                ;;

                        -*)
                                AC_MSG_CHECKING([whether pthreads work with $flag])
                                PTHREAD_CFLAGS="$flag"
                                ;;

                        *)
                                AC_MSG_CHECKING([for the pthreads library -l$flag])
                                PTHREAD_LIBS="-l$flag"
                                ;;
                        esac

                        save_LIBS="$LIBS"
                        save_CFLAGS="$CFLAGS"
                        LIBS="$PTHREAD_LIBS $LIBS"
                        CFLAGS="$CFLAGS $PTHREAD_CFLAGS"

                        # Check for various functions.  We must include pthread.h,
                        # since some functions may be macros.  (On the Sequent, we
                        # need a special flag -Kthread to make this header compile.)
                        # We check for pthread_join because it is in -lpthread on IRIX
                        # while pthread_create is in libc.  We check for pthread_attr_init
                        # due to DEC craziness with -lpthreads.  We check for
                        # pthread_cleanup_push because it is one of the few pthread
                        # functions on Solaris that doesn't have a non-functional libc stub.
                        # We try pthread_create on general principles.
                        AC_TRY_LINK([#include <pthread.h>], [
                                        pthread_t th; 
                                        pthread_join(th, 0);
                                        pthread_attr_init(0); 
                                        pthread_cleanup_push(0, 0);
                                        pthread_create(0,0,0,0); 
                                        pthread_cleanup_pop(0); 
                                ], [ax_pthread_ok=yes])

                        LIBS="$save_LIBS"
                        CFLAGS="$save_CFLAGS"

                        AC_MSG_RESULT($ax_pthread_ok)
                        if test "x$ax_pthread_ok" = xyes; then
                                break;
                        fi

                        PTHREAD_LIBS=""
                        PTHREAD_CFLAGS=""
                done
        fi

        # Various other checks:
        if test "x$ax_pthread_ok" = xyes; then
                save_LIBS="$LIBS"
                LIBS="$PTHREAD_LIBS $LIBS"
                save_CFLAGS="$CFLAGS"
                CFLAGS="$CFLAGS $PTHREAD_CFLAGS"

                # Detect AIX lossage: threads are created detached by default
                # and the JOINABLE attribute has a nonstandard name (UNDETACHED).
                AC_MSG_CHECKING([for joinable pthread attribute])
                AC_TRY_LINK([#include <pthread.h>], [
                                int attr=PTHREAD_CREATE_JOINABLE;
                        ], ok=PTHREAD_CREATE_JOINABLE, ok=unknown)
                
                if test x"$ok" = xunknown; then
                        AC_TRY_LINK([#include <pthread.h>], [
                                        int attr=PTHREAD_CREATE_UNDETACHED;
                                ], ok=PTHREAD_CREATE_UNDETACHED, ok=unknown)
                fi
                
                if test x"$ok" != xPTHREAD_CREATE_JOINABLE; then
                        AC_DEFINE(PTHREAD_CREATE_JOINABLE, $ok, 
                                [Define to the necessary symbol if this constant uses a non-standard name on your system.])
                fi
                
                AC_MSG_RESULT(${ok})
                if test x"$ok" = xunknown; then
                        AC_MSG_WARN([we do not know how to create joinable pthreads])
                fi

                AC_MSG_CHECKING([if more special flags are required for pthreads])
                
                flag=no
                case "${host_cpu}-${host_os}" in
                        *-aix* | *-freebsd*)     flag="-D_THREAD_SAFE";;
                        *solaris* | *-osf* | *-hpux* | *linux* | *linux-gnu*) flag="-D_REENTRANT";; 
                esac
                
                AC_MSG_RESULT(${flag})
                if test "x$flag" != xno; then
                        PTHREAD_CFLAGS=" $flag $PTHREAD_CFLAGS "
                fi

                LIBS="$save_LIBS"
                CFLAGS="$save_CFLAGS"

                # More AIX lossage: must compile with cc_r
                AC_CHECK_PROG(PTHREAD_CC, cc_r, cc_r, ${CC})
        else
                PTHREAD_CC="$CC"
        fi

        AC_SUBST(PTHREAD_LIBS)
        AC_SUBST(PTHREAD_CFLAGS)
        AC_SUBST(PTHREAD_CC)

        # Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
        if test x"$ax_pthread_ok" = xyes; then
                AC_DEFINE(HAVE_PTHREAD, 1, [Define if you have POSIX threads libraries and header files.])
                :
        fi

        AC_LANG_POP([C])
]) dnl AX_PTHREAD

#ifndef UMUQ_MACROS_H
#define UMUQ_MACROS_H

#define UMUQ_MAJOR_VERSION 1
#define UMUQ_MINOR_VERSION 0
#define UMUQ_REVISION_VERSION 0

#define UMUQ_VERSION_AT_LEAST(x, y, z) (UMUQ_MAJOR_VERSION > x || (UMUQ_MAJOR_VERSION >= x && (UMUQ_MINOR_VERSION > y || (UMUQ_MINOR_VERSION >= y && UMUQ_REVISION_VERSION >= z))))

// Operating system identification, UMUQ_OS_*

/// UMUQ_OS_UNIX set to 1 if the OS is a unix variant
#if defined(__unix__) || defined(__unix)
#define UMUQ_OS_UNIX 1
#else
#define UMUQ_OS_UNIX 0
#endif

/// UMUQ_OS_LINUX set to 1 if the OS is based on Linux kernel
#if defined(__linux__)
#define UMUQ_OS_LINUX 1
#else
#define UMUQ_OS_LINUX 0
#endif

/// UMUQ_OS_ANDROID set to 1 if the OS is Android
// note: ANDROID is defined when using ndk_build, __ANDROID__ is defined when using a standalone toolchain.
#if defined(__ANDROID__) || defined(ANDROID)
#define UMUQ_OS_ANDROID 1
#else
#define UMUQ_OS_ANDROID 0
#endif

/// UMUQ_OS_GNULINUX set to 1 if the OS is GNU Linux and not Linux-based OS (e.g., not android)
#if defined(__gnu_linux__) && !(UMUQ_OS_ANDROID)
#define UMUQ_OS_GNULINUX 1
#else
#define UMUQ_OS_GNULINUX 0
#endif

/// UMUQ_OS_BSD set to 1 if the OS is a BSD variant
#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__DragonFly__)
#define UMUQ_OS_BSD 1
#else
#define UMUQ_OS_BSD 0
#endif

/// UMUQ_OS_MAC set to 1 if the OS is MacOS
#if defined(__APPLE__)
#define UMUQ_OS_MAC 1
#else
#define UMUQ_OS_MAC 0
#endif

// A Clang feature extension to determine compiler features.
// We use it to determine 'cxx_rvalue_references'
#ifndef __has_feature
#define __has_feature(x) 0
#endif

/** Allows to disable some optimizations which might affect the accuracy of the result.
  * Such optimization are enabled by default, and set UMUQ_FAST_MATH to 0 to disable them.
  * They currently include:
  *   - single precision sin() and cos() for SSE and AVX vectorization.
  */
#ifndef UMUQ_FAST_MATH
#define UMUQ_FAST_MATH 1
#endif

#define UMUQ_DEBUG_VAR(x) std::cerr << #x << " = " << x << std::endl;

// concatenate two tokens
#define UMUQ_CAT2(a, b) a##b
#define UMUQ_CAT(a, b) UMUQ_CAT2(a, b)

// convert a token to a string
#define UMUQ_MAKESTRING2(a) #a
#define UMUQ_MAKESTRING(a) UMUQ_MAKESTRING2(a)

#ifdef NDEBUG
#ifndef UMUQ_NO_DEBUG
#define UMUQ_NO_DEBUG
#endif
#endif

// UMUQ_plain_assert is where we implement the workaround for the assert() bug in GCC <= 4.3, see bug 89
#ifdef UMUQ_NO_DEBUG
#define UMUQ_plain_assert(x)
#else
#include <cstdlib>  // for abort
#include <iostream> // for std::cerr
namespace UMUQ
{
bool copy_bool(bool b);
bool copy_bool(bool b) { return b; }

[[noreturn]] void assert_fail(const char *condition, const char *function, const char *file, int line);
[[noreturn]] void assert_fail(const char *condition, const char *function, const char *file, int line)
{
    std::cerr << "assertion failed: " << condition << " in function " << function << " at " << file << ":" << line << std::endl;
    std::abort();
}
} //namespace UMUQ
#define UMUQ_plain_assert(x)                                                                \
    do                                                                                      \
    {                                                                                       \
        if (!UMUQ::copy_bool(x))                                                            \
            UMUQ::assert_fail(UMUQ_MAKESTRING(x), __PRETTY_FUNCTION__, __FILE__, __LINE__); \
    } while (false)
#endif

// UMUQ_assert can be overridden
#ifndef UMUQ_assert
#define UMUQ_assert(x) UMUQ_plain_assert(x)
#endif

// this macro allows to get rid of linking errors about multiply defined functions.
//  - static is not very good because it prevents definitions from different object files to be merged.
//           So static causes the resulting linked executable to be bloated with multiple copies of the same function.
//  - inline is not perfect either as it unwantedly hints the compiler toward inlining the function.
#define UMUQ_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS UMUQ_DEVICE_FUNC
#define UMUQ_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS UMUQ_DEVICE_FUNC inline

#endif // UMUQ_MACROS_H

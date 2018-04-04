#ifndef UMHBM_MACROS_H
#define UMHBM_MACROS_H

#define UMHBM_WORLD_VERSION 1
#define UMHBM_MAJOR_VERSION 1
#define UMHBM_MINOR_VERSION 0

#define UMHBM_VERSION_AT_LEAST(x, y, z) (UMHBM_WORLD_VERSION > x || (UMHBM_WORLD_VERSION >= x && (UMHBM_MAJOR_VERSION > y || (UMHBM_MAJOR_VERSION >= y && UMHBM_MINOR_VERSION >= z))))

// Operating system identification, UMHBM_OS_*

/// UMHBM_OS_UNIX set to 1 if the OS is a unix variant
#if defined(__unix__) || defined(__unix)
#define UMHBM_OS_UNIX 1
#else
#define UMHBM_OS_UNIX 0
#endif

/// UMHBM_OS_LINUX set to 1 if the OS is based on Linux kernel
#if defined(__linux__)
#define UMHBM_OS_LINUX 1
#else
#define UMHBM_OS_LINUX 0
#endif

/// UMHBM_OS_ANDROID set to 1 if the OS is Android
// note: ANDROID is defined when using ndk_build, __ANDROID__ is defined when using a standalone toolchain.
#if defined(__ANDROID__) || defined(ANDROID)
#define UMHBM_OS_ANDROID 1
#else
#define UMHBM_OS_ANDROID 0
#endif

/// UMHBM_OS_GNULINUX set to 1 if the OS is GNU Linux and not Linux-based OS (e.g., not android)
#if defined(__gnu_linux__) && !(UMHBM_OS_ANDROID)
#define UMHBM_OS_GNULINUX 1
#else
#define UMHBM_OS_GNULINUX 0
#endif

/// UMHBM_OS_BSD set to 1 if the OS is a BSD variant
#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__DragonFly__)
#define UMHBM_OS_BSD 1
#else
#define UMHBM_OS_BSD 0
#endif

/// UMHBM_OS_MAC set to 1 if the OS is MacOS
#if defined(__APPLE__)
#define UMHBM_OS_MAC 1
#else
#define UMHBM_OS_MAC 0
#endif

// A Clang feature extension to determine compiler features.
// We use it to determine 'cxx_rvalue_references'
#ifndef __has_feature
#define __has_feature(x) 0
#endif

/** Allows to disable some optimizations which might affect the accuracy of the result.
  * Such optimization are enabled by default, and set UMHBM_FAST_MATH to 0 to disable them.
  * They currently include:
  *   - single precision ArrayBase::sin() and ArrayBase::cos() for SSE and AVX vectorization.
  */
#ifndef UMHBM_FAST_MATH
#define UMHBM_FAST_MATH 1
#endif

#define UMHBM_DEBUG_VAR(x) std::cerr << #x << " = " << x << std::endl;

// concatenate two tokens
#define UMHBM_CAT2(a, b) a##b
#define UMHBM_CAT(a, b) UMHBM_CAT2(a, b)

// convert a token to a string
#define UMHBM_MAKESTRING2(a) #a
#define UMHBM_MAKESTRING(a) UMHBM_MAKESTRING2(a)

#ifdef NDEBUG
#ifndef UMHBM_NO_DEBUG
#define UMHBM_NO_DEBUG
#endif
#endif

// UMHBM_plain_assert is where we implement the workaround for the assert() bug in GCC <= 4.3, see bug 89
#ifdef UMHBM_NO_DEBUG
#define UMHBM_plain_assert(x)
#else
#include <cstdlib>  // for abort
#include <iostream> // for std::cerr
bool copy_bool(bool b) { return b; }
inline void assert_fail(const char *condition, const char *function, const char *file, int line)
{
    std::cerr << "assertion failed: " << condition << " in function " << function << " at " << file << ":" << line << std::endl;
    abort();
}
#define UMHBM_plain_assert(x)                                                                           \
    do                                                                                                  \
    {                                                                                                   \
        if (!UMHBM::internal::copy_bool(x))                                                             \
            UMHBM::internal::assert_fail(UMHBM_MAKESTRING(x), __PRETTY_FUNCTION__, __FILE__, __LINE__); \
    } while (false)
#endif

// UMHBM_assert can be overridden
#ifndef UMHBM_assert
#define UMHBM_assert(x) UMHBM_plain_assert(x)
#endif

// this macro allows to get rid of linking errors about multiply defined functions.
//  - static is not very good because it prevents definitions from different object files to be merged.
//           So static causes the resulting linked executable to be bloated with multiple copies of the same function.
//  - inline is not perfect either as it unwantedly hints the compiler toward inlining the function.
#define UMHBM_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS UMHBM_DEVICE_FUNC
#define UMHBM_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS UMHBM_DEVICE_FUNC inline

#endif // UMHBM_MACROS_H

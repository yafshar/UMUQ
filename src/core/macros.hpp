#ifndef UMUQ_MACROS_H
#define UMUQ_MACROS_H

/*!
 * \brief Given a version number MAJOR.MINOR.PATCH, increment the:
 * \b MAJOR version when you make incompatible changes,
 * \b MINOR version when you add functionality in a backwards-compatible manner, and
 * \b PATCH version when you make backwards-compatible bug fixes. 
 * 
 * Ref:
 * https://semver.org
 */

#define UMUQ_MAJOR_VERSION 1
#define UMUQ_MINOR_VERSION 0
#define UMUQ_PATCH_VERSION 0
#define UMUQ_VERSION_AT_LEAST(x, y, z) (UMUQ_MAJOR_VERSION > x || (UMUQ_MAJOR_VERSION >= x && (UMUQ_MINOR_VERSION > y || (UMUQ_MINOR_VERSION >= y && UMUQ_PATCH_VERSION >= z))))

//! Operating system identification, UMUQ_OS_*

//! UMUQ_OS_UNIX set to 1 if the OS is a unix variant
#if defined(__unix__) || defined(__unix)
#define UMUQ_OS_UNIX 1
#else
#define UMUQ_OS_UNIX 0
#endif

//! UMUQ_OS_LINUX set to 1 if the OS is based on Linux kernel
#if defined(__linux__)
#define UMUQ_OS_LINUX 1
#else
#define UMUQ_OS_LINUX 0
#endif

//! UMUQ_OS_ANDROID set to 1 if the OS is Android
//! note: ANDROID is defined when using ndk_build, __ANDROID__ is defined when using a standalone toolchain.
#if defined(__ANDROID__) || defined(ANDROID)
#define UMUQ_OS_ANDROID 1
#else
#define UMUQ_OS_ANDROID 0
#endif

//! UMUQ_OS_GNULINUX set to 1 if the OS is GNU Linux and not Linux-based OS (e.g., not android)
#if defined(__gnu_linux__) && !(UMUQ_OS_ANDROID)
#define UMUQ_OS_GNULINUX 1
#else
#define UMUQ_OS_GNULINUX 0
#endif

//! UMUQ_OS_BSD set to 1 if the OS is a BSD variant
#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__DragonFly__)
#define UMUQ_OS_BSD 1
#else
#define UMUQ_OS_BSD 0
#endif

//! UMUQ_OS_MAC set to 1 if the OS is MacOS
#if defined(__APPLE__)
#define UMUQ_OS_MAC 1
#else
#define UMUQ_OS_MAC 0
#endif

//! A Clang feature extension to determine compiler features.
//! We use it to determine 'cxx_rvalue_references'
#ifndef __has_feature
#define __has_feature(x) 0
#endif

/*! 
 * Allows to disable some optimizations which might affect the accuracy of the result.
 * Such optimization are enabled by default, and set UMUQ_FAST_MATH to 0 to disable them.
 * They currently include:
 *  - single precision sin() and cos() for SSE and AVX vectorization.
 */
#ifndef UMUQ_FAST_MATH
#define UMUQ_FAST_MATH 1
#endif

#define UMUQ_DEBUG_VAR(x) std::cerr << #x << " = " << x << std::endl;

//! concatenate two tokens
#define UMUQ_CAT2(a, b) a##b
#define UMUQ_CAT(a, b) UMUQ_CAT2(a, b)

//! convert a token to a string
#define UMUQ_MAKESTRING2(a) #a
#define UMUQ_MAKESTRING(a) UMUQ_MAKESTRING2(a)

#ifdef NODEBUG
#ifndef UMUQ_NO_DEBUG
#define UMUQ_NO_DEBUG
#endif
#endif

//! UMUQ_plain_assert is where we implement the workaround for the assert() bug in GCC <= 4.3, see bug 89
#ifdef UMUQ_NO_DEBUG
#define UMUQ_plain_assert(x)
#else
#include <cstdlib>  // for abort
#include <iostream> // for std::cerr
namespace UMUQ
{
bool copy_bool(bool b);
bool copy_bool(bool b) { return b; }

[[noreturn]] void assert_fail(const char *condition, const char *function, const char *file, long line);
[[noreturn]] void assert_fail(const char *condition, const char *function, const char *file, long line) {
    std::cerr << "Assertion failed: " << condition << " in function " << function << " at " << file << ":" << line << std::endl;
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

#include <sstream>
namespace UMUQ
{
namespace internal
{
/*!
 * \brief This function formats messages, filename, line number and function name into an std::ostringstream object
 * 
 * \param message1      Starting message 
 * \param fileName      File name
 * \param lineNumber    Line number
 * \param functionName  Function name    
 * \param message2      Ending message
 * 
 * \returns The combined std::ostringstream object as a string
 */
std::string FormatMessageFileLineFunctionMessage(std::string const &message1,
                                                 std::string const &fileName,
                                                 long lineNumber,
                                                 std::string const &functionName,
                                                 std::string const &message2)
{
    std::ostringstream ss;
    ss << "\n";
    ss << message1 << ":" << fileName << ":" << lineNumber << ":@(" << functionName << ")\n";
    ss << message2 << "\n\n";
    return ss.str();
}
} // namespace internal
} // namespace UMUQ

#define UMUQFAIL(msg)                                                                                                                 \
    std::ostringstream ss;                                                                                                            \
    ss << msg;                                                                                                                        \
    std::string _Message_(UMUQ::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
    std::cerr << _Message_;                                                                                                           \
    throw(std::runtime_error(ss.str()));

#define UMUQFAILRETURN(msg)                                                                                                           \
    std::ostringstream ss;                                                                                                            \
    ss << msg;                                                                                                                        \
    std::string _Message_(UMUQ::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
    std::cerr << _Message_;                                                                                                           \
    return false;

#define UMUQFAILRETURNNULL(msg)                                                                                                       \
    std::ostringstream ss;                                                                                                            \
    ss << msg;                                                                                                                        \
    std::string _Message_(UMUQ::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
    std::cerr << _Message_;                                                                                                           \
    return nullptr;

#define UMUQFAILRETURNSTRING(msg)                                                                                                     \
    std::ostringstream ss;                                                                                                            \
    ss << msg;                                                                                                                        \
    std::string _Message_(UMUQ::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
    std::cerr << _Message_;                                                                                                           \
    return ss.str();

#define UMUQFAILS(msg, index)                                                                                                                \
    std::ostringstream ss;                                                                                                                   \
    ss << msg;                                                                                                                               \
    std::string _Message_##index(UMUQ::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
    std::cerr << _Message_##index;                                                                                                           \
    throw(std::runtime_error(ss.str()));

#define UMUQWARNING(msg)                                                                                                                \
    std::ostringstream ss;                                                                                                              \
    ss << msg;                                                                                                                          \
    std::string _Message_(UMUQ::internal::FormatMessageFileLineFunctionMessage("Warning", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
    std::cerr << _Message_;

#define UMUQWARNINGS(msg, index)                                                                                                               \
    std::ostringstream ss;                                                                                                                     \
    ss << msg;                                                                                                                                 \
    std::string _Message_##index(UMUQ::internal::FormatMessageFileLineFunctionMessage("Warning", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
    std::cerr << _Message_##index;

#define UMUQASSERT(condition, msg) \
    if (!(condition))              \
    UMUQFAIL(msg)

#define UMUQASSERTS(condition, msg, index) \
    if (!(condition))                      \
    UMUQFAIL(msg, index)

#endif // UMUQ_MACROS_H

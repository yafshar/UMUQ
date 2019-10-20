#ifndef UMUQ_MACROS_H
#define UMUQ_MACROS_H

/*!
 * \ingroup Core_Module
 *
 * \brief Given a version number MAJOR.MINOR.PATCH, increment the:<br>
 * - \b MAJOR version when you make incompatible changes,
 * - \b MINOR version when you add functionality in a backwards-compatible manner, and
 * - \b PATCH version when you make backwards-compatible bug fixes.
 *
 * Reference:<br>
 * https://semver.org
 */
#define UMUQ_MAJOR_VERSION 1
#define UMUQ_MINOR_VERSION 0
#define UMUQ_PATCH_VERSION 0
#define UMUQ_VERSION_AT_LEAST(x, y, z) (UMUQ_MAJOR_VERSION > x || (UMUQ_MAJOR_VERSION >= x && (UMUQ_MINOR_VERSION > y || (UMUQ_MINOR_VERSION >= y && UMUQ_PATCH_VERSION >= z))))

/*!
 * \ingroup Core_Module
 *
 * \brief Operating system identification, \c UMUQ_OS_*
 *
 * \c UMUQ_OS_UNIX set to 1 if the OS is a unix variant
 */
#if defined(__unix__) || defined(__unix)
#define UMUQ_OS_UNIX 1
#else
#define UMUQ_OS_UNIX 0
#endif

/*!
 * \ingroup Core_Module
 *
 * \brief Operating system identification, \c UMUQ_OS_*
 *
 * \c UMUQ_OS_LINUX set to 1 if the OS is based on Linux kernel
 */
#if defined(__linux__)
#define UMUQ_OS_LINUX 1
#else
#define UMUQ_OS_LINUX 0
#endif

/*!
 * \ingroup Core_Module
 *
 * \brief Operating system identification, \c UMUQ_OS_*
 *
 * \c UMUQ_OS_ANDROID set to 1 if the OS is Android
 *
 */
#if defined(__ANDROID__) || defined(ANDROID)
#define UMUQ_OS_ANDROID 1
#else
#define UMUQ_OS_ANDROID 0
#endif

/*!
 * \ingroup Core_Module
 *
 * \brief Operating system identification, \c UMUQ_OS_*
 *
 * \c UMUQ_OS_GNULINUX set to 1 if the OS is GNU Linux and not Linux-based OS (e.g., not android)
 */
#if defined(__gnu_linux__) && !(UMUQ_OS_ANDROID)
#define UMUQ_OS_GNULINUX 1
#else
#define UMUQ_OS_GNULINUX 0
#endif

/*!
 * \ingroup Core_Module
 *
 * \brief Operating system identification, \c UMUQ_OS_*
 *
 * \c UMUQ_OS_BSD set to 1 if the OS is a BSD variant
 */
#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__DragonFly__)
#define UMUQ_OS_BSD 1
#else
#define UMUQ_OS_BSD 0
#endif

/*!
 * \ingroup Core_Module
 *
 * \brief Operating system identification, \c UMUQ_OS_*
 *
 * \c UMUQ_OS_MAC set to 1 if the OS is MacOS
 */
#if defined(__APPLE__)
#define UMUQ_OS_MAC 1
#else
#define UMUQ_OS_MAC 0
#endif

/*!
 * \ingroup Core_Module
 *
 * A Clang feature extension to determine compiler features.<br>
 * We use it to determine 'cxx_rvalue_references'
 */
#ifndef __has_feature
#define __has_feature(x) 0
#endif

/*!
 * \ingroup Core_Module
 *
 * Allows to disable some optimizations which might affect the accuracy of the result.<br>
 * Such optimization are enabled by default, and set UMUQ_FAST_MATH to 0 to disable them.<br>
 * They currently include:<br>
 * - single precision \c sin() and \c cos() for SSE and AVX vectorization.
 */
#ifndef UMUQ_FAST_MATH
#define UMUQ_FAST_MATH 1
#endif

/*!
 * \ingroup Core_Module
 */
#define UMUQ_DEBUG_VAR(x) std::cerr << #x << " = " << x << std::endl;

/*!
 * \ingroup Core_Module
 *
 * Concatenate two tokens
 */
#define UMUQ_CAT2(a, b) a##b

/*!
 * \ingroup Core_Module
 *
 * Concatenate three tokens
 */
#define UMUQ_CAT3(a, b, c) a##b##c

/*!
 * \ingroup Core_Module
 *
 * Concatenate four tokens
 */
#define UMUQ_CAT4(a, b, c, d) a##b##c##d

/*!
 * \ingroup Core_Module
 *
 * Concatenate five tokens
 */
#define UMUQ_CAT5(a, b, c, d, e) a##b##c##d##e

/*!
 * \ingroup Core_Module
 *
 * Concatenate six tokens
 */
#define UMUQ_CAT6(a, b, c, d, e, f) a##b##c##d##e##f

/*!
 * \ingroup Core_Module
 *
 * Convert a token to a string
 */
#define UMUQ_MAKESTRING2(a) #a

/*!
 * \ingroup Core_Module
 *
 * Convert a token to a string
 */
#define UMUQ_MAKESTRING(a) UMUQ_MAKESTRING2(a)

/*!
 * \ingroup Core_Module
 *
 * \brief A macro to select the appropriate override.
 *
 * A macro that uses the [paired, sliding arg list](https://gist.github.com/dhh1128/0cf088f4f681f619b051) technique to select the appropriate override.
 * It supports the set of 1 to 6 args.
 */
#define UMUQ_OVERRIDE(_1, _2, _3, _4, _5, _6, NAME, ...) NAME

/*!
 * \ingroup Core_Module
 *
 * \brief A macro that concatenates strings
 *
 * It is macro that concatenates either 5, 4, 3 or 2 strings together.
 */
#define UMUQ_CAT(...)                                                                 \
    UMUQ_OVERRIDE(__VA_ARGS__, UMUQ_CAT6, UMUQ_CAT5, UMUQ_CAT4, UMUQ_CAT3, UMUQ_CAT2) \
    (__VA_ARGS__)

/*!
 * \ingroup Core_Module
 *
 * No debug token
 */
#ifdef NODEBUG
#ifndef UMUQ_NO_DEBUG
#define UMUQ_NO_DEBUG
#endif
#endif

/*!
 * \ingroup Core_Module
 *
 */
#ifdef UMUQ_NO_DEBUG
/*!
 * \ingroup Core_Module
 *
 * \brief Assertion
 */
#define UMUQ_plain_assert(x)
#else
#include <cstdlib>  // for abort
#include <iostream> // for std::cerr

namespace umuq
{
/*!
 * \ingroup Core_Module
 *
 * \brief Copy bool
 *
 * \param b Input logical
 */
bool copy_bool(bool b);
bool copy_bool(bool b) { return b; }

/*!
 * \ingroup Core_Module
 *
 * \brief Assertion
 *
 * \param condition Assertion condition
 * \param function  Function name
 * \param file      File name
 * \param line      Line number
 */
[[noreturn]] void assert_fail(const char *condition, const char *function, const char *file, long line);
[[noreturn]] void assert_fail(const char *condition, const char *function, const char *file, long line) {
    std::cerr << "Assertion failed: " << condition << " in function " << function << " at " << file << ":" << line << std::endl;
    std::abort();
}

} //namespace umuq

/*!
 * \ingroup Core_Module
 *
 * \brief Assertion
 *
 */
#define UMUQ_plain_assert(x)                                                                \
    do                                                                                      \
    {                                                                                       \
        if (!umuq::copy_bool(x))                                                            \
            umuq::assert_fail(UMUQ_MAKESTRING(x), __PRETTY_FUNCTION__, __FILE__, __LINE__); \
    } while (false)
#endif

/*!
 * \ingroup Core_Module
 *
 * \brief Assertion macro
 *
 * \c UMUQ_assert can be overridden
 */
#ifndef UMUQ_assert
#define UMUQ_assert(x) UMUQ_plain_assert(x)
#endif

#include <sstream>
namespace umuq
{

namespace internal
{
/*!
 * \ingroup Core_Module
 *
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
                                                 std::string const &message2);

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

#if HAVE_MPI == 1
/*!
 * \ingroup Core_Module
 *
 * \brief MPI safe call
 *
 * \param comm          MPI communicator
 * \param errorCode     Error code
 * \returns std::string Error message from input Error code
 */
std::string MPIErrorMessage(MPI_Comm const &comm, int const errorCode)
{
    int nrank;
    int nsize;
    int stringLength;
    int errorClass;
    char errorStringClass[MPI_MAX_ERROR_STRING];
    char errorStringCode[MPI_MAX_ERROR_STRING];
    char commName[MPI_MAX_OBJECT_NAME];

    MPI_Comm_get_name(comm, commName, &stringLength);
    MPI_Error_string(errorCode, errorStringCode, &stringLength);
    MPI_Error_class(errorCode, &errorClass);
    MPI_Error_string(errorClass, errorStringClass, &stringLength);

    MPI_Comm_rank(comm, &nrank);
    MPI_Comm_size(comm, &nsize);

    std::ostringstream ss;
    ss << "\n";
    ss << "[" << nrank << ":" << nsize << "]: [" << commName << "] :" << errorStringClass << "\n";
    ss << "[" << nrank << ":" << nsize << "]: [" << commName << "] :" << errorStringCode << "\n";
    return ss.str();
}

/*!
 * \ingroup Core_Module
 *
 * \brief  MPI safe call
 *
 * \param errorCode      Error code
 * \returns std::string  Error message from input Error code
 */
std::string MPIErrorMessage(int const errorCode)
{
    return MPIErrorMessage(MPI_COMM_WORLD, errorCode);
}
#endif // MPI

} // namespace internal
} // namespace umuq

#if HAVE_MPI == 1
/*!
 * \ingroup Core_Module
 *
 * \brief Terminates the execution environment
 *
 */
#define UMUQABORT(ss)              \
    MPI_Abort(MPI_COMM_WORLD, -1); \
    throw(std::runtime_error(ss.str()));
#else
#define UMUQABORT(ss) throw(std::runtime_error(ss.str()));
#endif // MPI

/*!
 * \ingroup Core_Module
 *
 * \brief Helper macro for printing one error message
 *
 * \warning
 * - This is not a stand alone macro to use!
 * - Use \c UMUQFAIL \sa UMUQFAIL
 */
#define UMUQFAIL_1MSG(msg1)                                                                                                                \
    {                                                                                                                                      \
        std::ostringstream ss;                                                                                                             \
        ss << msg1;                                                                                                                        \
        std::string _Messagef_(umuq::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
        std::cerr << _Messagef_;

/*!
 * \ingroup Core_Module
 *
 * \brief Helper macro for printing two combined error messages
 *
 * \warning
 * - This is not a stand alone macro to use!
 * - Use \c UMUQFAIL \sa UMUQFAIL
 */
#define UMUQFAIL_2MSG(msg1, msg2)                                                                                                          \
    {                                                                                                                                      \
        std::ostringstream ss;                                                                                                             \
        ss << msg1 << msg2;                                                                                                                \
        std::string _Messagef_(umuq::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
        std::cerr << _Messagef_;

/*!
 * \ingroup Core_Module
 *
 * \brief Helper macro for printing three combined error messages
 *
 * \warning
 * - This is not a stand alone macro to use!
 * - Use \c UMUQFAIL \sa UMUQFAIL
 */
#define UMUQFAIL_3MSG(msg1, msg2, msg3)                                                                                                    \
    {                                                                                                                                      \
        std::ostringstream ss;                                                                                                             \
        ss << msg1 << msg2 << msg3;                                                                                                        \
        std::string _Messagef_(umuq::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
        std::cerr << _Messagef_;

/*!
 * \ingroup Core_Module
 *
 * \brief Helper macro for printing four combined error messages
 *
 * \warning
 * - This is not a stand alone macro to use!
 * - Use \c UMUQFAIL \sa UMUQFAIL
 */
#define UMUQFAIL_4MSG(msg1, msg2, msg3, msg4)                                                                                              \
    {                                                                                                                                      \
        std::ostringstream ss;                                                                                                             \
        ss << msg1 << msg2 << msg3 << msg4;                                                                                                \
        std::string _Messagef_(umuq::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
        std::cerr << _Messagef_;

/*!
 * \ingroup Core_Module
 *
 * \brief Helper macro for printing five combined error messages
 *
 * \warning
 * - This is not a stand alone macro to use!
 * - Use \c UMUQFAIL \sa UMUQFAIL
 */
#define UMUQFAIL_5MSG(msg1, msg2, msg3, msg4, msg5)                                                                                        \
    {                                                                                                                                      \
        std::ostringstream ss;                                                                                                             \
        ss << msg1 << msg2 << msg3 << msg4 << msg5;                                                                                        \
        std::string _Messagef_(umuq::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
        std::cerr << _Messagef_;

/*!
 * \ingroup Core_Module
 *
 * \brief Helper macro for printing six combined error messages
 *
 * \warning
 * - This is not a stand alone macro to use!
 * - Use \c UMUQFAIL \sa UMUQFAIL
 */
#define UMUQFAIL_6MSG(msg1, msg2, msg3, msg4, msg5, msg6)                                                                                  \
    {                                                                                                                                      \
        std::ostringstream ss;                                                                                                             \
        ss << msg1 << msg2 << msg3 << msg4 << msg5 << msg6;                                                                                \
        std::string _Messagef_(umuq::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
        std::cerr << _Messagef_;

/*!
 * \ingroup Core_Module
 *
 * \brief Helper macro for printing one warning message
 *
 * \warning
 * - Use \c UMUQWARNING \sa UMUQWARNING
 */
#define UMUQWARNING_1MSG(msg1)                                                                                                               \
    {                                                                                                                                        \
        std::ostringstream ss;                                                                                                               \
        ss << msg1;                                                                                                                          \
        std::string _Messagew_(umuq::internal::FormatMessageFileLineFunctionMessage("Warning", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
        std::cerr << _Messagew_;                                                                                                             \
    }

/*!
 * \ingroup Core_Module
 *
 * \brief Helper macro for printing two combined warning messages
 *
 * \warning
 * - Use \c UMUQWARNING \sa UMUQWARNING
 */
#define UMUQWARNING_2MSG(msg1, msg2)                                                                                                         \
    {                                                                                                                                        \
        std::ostringstream ss;                                                                                                               \
        ss << msg1 << msg2;                                                                                                                  \
        std::string _Messagew_(umuq::internal::FormatMessageFileLineFunctionMessage("Warning", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
        std::cerr << _Messagew_;                                                                                                             \
    }

/*!
 * \ingroup Core_Module
 *
 * \brief Helper macro for printing three combined warning messages
 *
 * \warning
 * - Use \c UMUQWARNING \sa UMUQWARNING
 */
#define UMUQWARNING_3MSG(msg1, msg2, msg3)                                                                                                   \
    {                                                                                                                                        \
        std::ostringstream ss;                                                                                                               \
        ss << msg1 << msg2 << msg3;                                                                                                          \
        std::string _Messagew_(umuq::internal::FormatMessageFileLineFunctionMessage("Warning", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
        std::cerr << _Messagew_;                                                                                                             \
    }

/*!
 * \ingroup Core_Module
 *
 * \brief Helper macro for printing four combined warning messages
 *
 * \warning
 * - Use \c UMUQWARNING \sa UMUQWARNING
 */
#define UMUQWARNING_4MSG(msg1, msg2, msg3, msg4)                                                                                             \
    {                                                                                                                                        \
        std::ostringstream ss;                                                                                                               \
        ss << msg1 << msg2 << msg3 << msg4;                                                                                                  \
        std::string _Messagew_(umuq::internal::FormatMessageFileLineFunctionMessage("Warning", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
        std::cerr << _Messagew_;                                                                                                             \
    }

/*!
 * \ingroup Core_Module
 *
 * \brief Helper macro for printing five combined warning messages
 *
 * \warning
 * - Use \c UMUQWARNING \sa UMUQWARNING
 */
#define UMUQWARNING_5MSG(msg1, msg2, msg3, msg4, msg5)                                                                                       \
    {                                                                                                                                        \
        std::ostringstream ss;                                                                                                               \
        ss << msg1 << msg2 << msg3 << msg4 << msg5;                                                                                          \
        std::string _Messagew_(umuq::internal::FormatMessageFileLineFunctionMessage("Warning", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
        std::cerr << _Messagew_;                                                                                                             \
    }

/*!
 * \ingroup Core_Module
 *
 * \brief Helper macro for printing six combined warning messages
 *
 * \warning
 * - Use \c UMUQWARNING \sa UMUQWARNING
 */
#define UMUQWARNING_6MSG(msg1, msg2, msg3, msg4, msg5, msg6)                                                                                 \
    {                                                                                                                                        \
        std::ostringstream ss;                                                                                                               \
        ss << msg1 << msg2 << msg3 << msg4 << msg5 << msg6;                                                                                  \
        std::string _Messagew_(umuq::internal::FormatMessageFileLineFunctionMessage("Warning", __FILE__, __LINE__, __FUNCTION__, ss.str())); \
        std::cerr << _Messagew_;                                                                                                             \
    }

/*!
 * \ingroup Core_Module
 *
 * \brief Prints one (or up to 6 combined) message(s) and terminates the execution environment
 *
 */
#define UMUQFAIL(...)                                                                                                    \
    UMUQ_OVERRIDE(__VA_ARGS__, UMUQFAIL_6MSG, UMUQFAIL_5MSG, UMUQFAIL_4MSG, UMUQFAIL_3MSG, UMUQFAIL_2MSG, UMUQFAIL_1MSG) \
    (__VA_ARGS__)                                                                                                        \
        UMUQABORT(ss)                                                                                                    \
    }

/*!
 * \ingroup Core_Module
 *
 * \brief Prints one (or up to 6 combined) message(s) and return back as false
 *
 */
#define UMUQFAILRETURN(...)                                                                                              \
    UMUQ_OVERRIDE(__VA_ARGS__, UMUQFAIL_6MSG, UMUQFAIL_5MSG, UMUQFAIL_4MSG, UMUQFAIL_3MSG, UMUQFAIL_2MSG, UMUQFAIL_1MSG) \
    (__VA_ARGS__) return false;                                                                                          \
    }

/*!
 * \ingroup Core_Module
 *
 * \brief Prints one (or up to 6 combined) message(s) and return back as nullptr
 *
 */
#define UMUQFAILRETURNNULL(...)                                                                                          \
    UMUQ_OVERRIDE(__VA_ARGS__, UMUQFAIL_6MSG, UMUQFAIL_5MSG, UMUQFAIL_4MSG, UMUQFAIL_3MSG, UMUQFAIL_2MSG, UMUQFAIL_1MSG) \
    (__VA_ARGS__) return nullptr;                                                                                        \
    }

/*!
 * \ingroup Core_Module
 *
 * \brief Prints one (or up to 6 combined) message(s) and return back the failing string
 *
 */
#define UMUQFAILRETURNSTRING(...)                                                                                        \
    UMUQ_OVERRIDE(__VA_ARGS__, UMUQFAIL_6MSG, UMUQFAIL_5MSG, UMUQFAIL_4MSG, UMUQFAIL_3MSG, UMUQFAIL_2MSG, UMUQFAIL_1MSG) \
    (__VA_ARGS__) return ss.str();                                                                                       \
    }

/*!
 * \ingroup Core_Module
 *
 * \brief Prints one (or up to 6 combined) warning-message(s)
 *
 */
#define UMUQWARNING(...)                                                                                                                   \
    UMUQ_OVERRIDE(__VA_ARGS__, UMUQWARNING_6MSG, UMUQWARNING_5MSG, UMUQWARNING_4MSG, UMUQWARNING_3MSG, UMUQWARNING_2MSG, UMUQWARNING_1MSG) \
    (__VA_ARGS__)

/*!
 * \ingroup Core_Module
 *
 * \brief Asserts the condition and in case of failure prints one (or up to 6 combined) message(s) and terminates the execution environment
 *
 */
#define UMUQASSERT(condition, ...)                                                                                           \
    if (!(condition))                                                                                                        \
    {                                                                                                                        \
        UMUQ_OVERRIDE(__VA_ARGS__, UMUQFAIL_6MSG, UMUQFAIL_5MSG, UMUQFAIL_4MSG, UMUQFAIL_3MSG, UMUQFAIL_2MSG, UMUQFAIL_1MSG) \
        (__VA_ARGS__)                                                                                                        \
            UMUQABORT(ss)                                                                                                    \
    }                                                                                                                        \
    }

#if HAVE_MPI == 1
/*!
 * \ingroup Core_Module
 *
 * \brief Simple wrapper for safe MPI call, in case of a failure, it terminates the execution environment
 *
 */
#define UMUQMPI(MPIcall)                                                            \
    {                                                                               \
        int err = MPIcall;                                                          \
        if (err != MPI_SUCCESS)                                                     \
        {                                                                           \
            std::string msg = umuq::internal::MPIErrorMessage(MPI_COMM_WORLD, err); \
            UMUQFAIL(msg);                                                          \
        }                                                                           \
    }

#endif // MPI

#endif // UMUQ_MACROS

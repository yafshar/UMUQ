#ifndef UMUQ_MACROS_H
#define UMUQ_MACROS_H

/*!
 * \ingroup Core_Module
 * 
 * \brief Given a version number MAJOR.MINOR.PATCH, increment the:
 * 
 * - \b MAJOR version when you make incompatible changes,
 * - \b MINOR version when you add functionality in a backwards-compatible manner, and
 * - \b PATCH version when you make backwards-compatible bug fixes. 
 * 
 * Ref:
 * https://semver.org
 */
#define UMUQ_MAJOR_VERSION 1
#define UMUQ_MINOR_VERSION 0
#define UMUQ_PATCH_VERSION 0
#define UMUQ_VERSION_AT_LEAST(x, y, z) (UMUQ_MAJOR_VERSION > x || (UMUQ_MAJOR_VERSION >= x && (UMUQ_MINOR_VERSION > y || (UMUQ_MINOR_VERSION >= y && UMUQ_PATCH_VERSION >= z))))

/*!
 * \ingroup Core_Module
 * 
 * \brief Operating system identification, UMUQ_OS_*
 * UMUQ_OS_UNIX set to 1 if the OS is a unix variant
 */
#if defined(__unix__) || defined(__unix)
#define UMUQ_OS_UNIX 1
#else
#define UMUQ_OS_UNIX 0
#endif

/*!
 * \ingroup Core_Module
 * 
 * \brief Operating system identification, UMUQ_OS_*
 * UMUQ_OS_LINUX set to 1 if the OS is based on Linux kernel
 */
#if defined(__linux__)
#define UMUQ_OS_LINUX 1
#else
#define UMUQ_OS_LINUX 0
#endif

/*!
 * \ingroup Core_Module
 * 
 * \brief Operating system identification, UMUQ_OS_*
 * UMUQ_OS_ANDROID set to 1 if the OS is Android
 * NOTE:
 * ANDROID is defined when using ndk_build, __ANDROID__ is defined when using a standalone toolchain.
 */
#if defined(__ANDROID__) || defined(ANDROID)
#define UMUQ_OS_ANDROID 1
#else
#define UMUQ_OS_ANDROID 0
#endif

/*!
 * \ingroup Core_Module
 * 
 * \brief Operating system identification, UMUQ_OS_*
 *  UMUQ_OS_GNULINUX set to 1 if the OS is GNU Linux and not Linux-based OS (e.g., not android)
 */
#if defined(__gnu_linux__) && !(UMUQ_OS_ANDROID)
#define UMUQ_OS_GNULINUX 1
#else
#define UMUQ_OS_GNULINUX 0
#endif

/*!
 * \ingroup Core_Module
 * 
 * \brief Operating system identification, UMUQ_OS_*
 *  UMUQ_OS_BSD set to 1 if the OS is a BSD variant
 */
#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__DragonFly__)
#define UMUQ_OS_BSD 1
#else
#define UMUQ_OS_BSD 0
#endif

/*!
 * \ingroup Core_Module
 * 
 * \brief Operating system identification, UMUQ_OS_*
 *  UMUQ_OS_MAC set to 1 if the OS is MacOS
 */
#if defined(__APPLE__)
#define UMUQ_OS_MAC 1
#else
#define UMUQ_OS_MAC 0
#endif

/*!
 * \ingroup Core_Module
 * 
 * A Clang feature extension to determine compiler features.
 *  We use it to determine 'cxx_rvalue_references'
 */
#ifndef __has_feature
#define __has_feature(x) 0
#endif

/*! 
 * \ingroup Core_Module
 * 
 * Allows to disable some optimizations which might affect the accuracy of the result.
 * Such optimization are enabled by default, and set UMUQ_FAST_MATH to 0 to disable them.
 * They currently include:
 *  - single precision sin() and cos() for SSE and AVX vectorization.
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
#define UMUQ_CAT(a, b) UMUQ_CAT2(a, b)

/*! 
 * \ingroup Core_Module
 * 
 * Convert a token to a string
 */
#define UMUQ_MAKESTRING2(a) #a
#define UMUQ_MAKESTRING(a) UMUQ_MAKESTRING2(a)

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

// UMUQ_assert can be overridden
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
 * \brief Prints the failing message and terminates the execution environment
 * 
 */
#define UMUQFAIL(msg)                                                                                                                   \
    std::ostringstream ssf;                                                                                                             \
    ssf << msg;                                                                                                                         \
    std::string _Messagef_(umuq::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ssf.str())); \
    std::cerr << _Messagef_;                                                                                                            \
    UMUQABORT(ssf)

/*!
 * \ingroup Core_Module
 * 
 * \brief Prints the failing message and return back as false
 * 
 */
#define UMUQFAILRETURN(msg)                                                                                                               \
    std::ostringstream ssfr;                                                                                                              \
    ssfr << msg;                                                                                                                          \
    std::string _Messagefr_(umuq::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ssfr.str())); \
    std::cerr << _Messagefr_;                                                                                                             \
    return false;

/*!
 * \ingroup Core_Module
 * 
 * \brief Prints the failing message and return back as nullptr
 * 
 */
#define UMUQFAILRETURNNULL(msg)                                                                                                             \
    std::ostringstream ssfrn;                                                                                                               \
    ssfrn << msg;                                                                                                                           \
    std::string _Messagefrn_(umuq::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ssfrn.str())); \
    std::cerr << _Messagefrn_;                                                                                                              \
    return nullptr;

/*!
 * \ingroup Core_Module
 * 
 * \brief Prints the failing message and return back the failing message string
 * 
 */
#define UMUQFAILRETURNSTRING(msg)                                                                                                        \
    std::ostringstream ssfrs;                                                                                                            \
    ssfrs << msg;                                                                                                                        \
    std::string _Message_(umuq::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ssfrs.str())); \
    std::cerr << _Message_;                                                                                                              \
    return ssfrs.str();

/*!
 * \ingroup Core_Module
 * 
 * \brief Prints the failing message with the index number and terminates the execution environment
 * 
 */
#define UMUQFAILS(msg, index)                                                                                                                       \
    std::ostringstream ss##index;                                                                                                                   \
    ss##index << msg;                                                                                                                               \
    std::string _Message_##index(umuq::internal::FormatMessageFileLineFunctionMessage("Error", __FILE__, __LINE__, __FUNCTION__, ss##index.str())); \
    std::cerr << _Message_##index;                                                                                                                  \
    UMUQABORT(ss##index)

/*!
 * \ingroup Core_Module
 * 
 * \brief Prints the warning message
 * 
 */
#define UMUQWARNING(msg)                                                                                                                  \
    std::ostringstream ssw;                                                                                                               \
    ssw << msg;                                                                                                                           \
    std::string _Messagew_(umuq::internal::FormatMessageFileLineFunctionMessage("Warning", __FILE__, __LINE__, __FUNCTION__, ssw.str())); \
    std::cerr << _Messagew_;

/*!
 * \ingroup Core_Module
 * 
 * \brief Prints the warning message with the index number 
 * 
 */
#define UMUQWARNINGS(msg, index)                                                                                                                       \
    std::ostringstream ssw##index;                                                                                                                     \
    ssw##index << msg;                                                                                                                                 \
    std::string _Message_##index(umuq::internal::FormatMessageFileLineFunctionMessage("Warning", __FILE__, __LINE__, __FUNCTION__, ssw##index.str())); \
    std::cerr << _Message_##index;

/*!
 * \ingroup Core_Module
 * 
 * \brief Asserts the condition and in case of failure prints the failing message and terminates the execution environment
 * 
 */
#define UMUQASSERT(condition, msg) \
    if (!(condition))              \
    UMUQFAIL(msg)

/*!
 * \ingroup Core_Module
 * 
 * \brief Asserts the condition and in case of failure prints the failing message with the index number and terminates the execution environment
 * 
 */
#define UMUQASSERTS(condition, msg, index) \
    if (!(condition))                      \
    UMUQFAILS(msg, index)

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

#ifndef UMUQ_UTILITY_H
#define UMUQ_UTILITY_H

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#else
#if _XOPEN_SOURCE < 500
#undef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif
#endif
#ifndef _BSD_SOURCE
#define _BSD_SOURCE 1
#endif

#include <unistd.h>    // fork, execvp, chdir
#include <sys/types.h> // waitpid
#include <sys/wait.h>  // waitpid

namespace umuq
{

/*! \class utility
 *
 * \brief This class includes some utility helpers.
 *	
 * Utility class contains functionality for exectuing commands
 * 
 */
class utility
{
  public:
    /*!
     * \brief Executing command from a spawner
     * 
     * \c fork and \c execvp in this function are used in sequence to get a new program running as a child 
     * of a current process (spawner with \c spawnerId ID).
     * 
     * \b fork() : <br>
     * After the system call to \c fork(), Unix will make an exact copy of the 
     * parent's address space and give it to the child. <br>
     * Therefore, the parent and child processes have separate address spaces. <br>
     * \c fork() returns a positive value, the process ID of the child process, to the parent. 
     * If \c fork() returns a negative value, the creation of a child process was unsuccessful
     * otherwise, \c fork() returns a zero to the newly created child process.
     * 
     * \b chdir() : <br>
     * The \c chdir() function shall cause the directory named by the pathname pointed 
     * to by the path argument to become the current working directory.<br>
     * 
     * Reference: <br>
     * https://linux.die.net/man/3/chdir
     *
     * \b execvp() : <br>
     * The \c execvp() system call requires two arguments:
     * - The first argument is a character string that contains the name of a file to be executed. <br>
     * - The second argument is a pointer to an array of character strings. More precisely, its type is char **, 
     *   which is exactly identical to the argv array used in the main program.
     * 
     * When \c execvp() is executed, the program file given by the first argument will be loaded into the caller's 
     * address space and over-write the program there. Then, the second argument will be provided to the program 
     * and starts the execution. As a result, once the specified program file starts its execution, the original 
     * program in the caller's address space is gone and is replaced by the new program. <br>
     * It returns a negative value if the execution fails (e.g., the request file does not exist). 
     * 
     * Reference: <br>
     * https://linux.die.net/man/3/execvp
     *
     * \b waitpid() : <br> 
     * The \c waitpid() system call suspends execution of the calling process until a child specified by pid argument 
     * has changed state.<br>
     * 0 means wait for any child process whose process group ID is equal to that of the calling process
     * 
     * Reference: <br> 
     * https://linux.die.net/man/2/waitpid
     *
     * 
     * \param spawnerId  Id of a spawner
     * \param argv       A pointer to an array of character strings. More precisely, its type is char **, 
     *                   which is identical to the argv array used in the main program
     * \param dirPath    Directory PATH in which to execute commands
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool executeCommand(int const spawnerId, char **argv, const char *dirPath = nullptr);

    /*!
     * \brief Calls the host environment's command processor (e.g. /bin/sh, cmd.exe, command.com) with 
     * the parameter command
     * 
     * \param inCommand  Input command, character string identifying the command to be run in the command 
     *                   processor. If a null pointer is given, command processor is checked for existence 
     * \param dirPath    Directory PATH in which to execute command
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool executeCommand(std::string const &inCommand, std::string const &dirPath = std::string());
};

bool utility::executeCommand(int const spawnerId, char **argv, const char *dirPath)
{
    pid_t pid = fork();

    /*! If \c fork() returns a negative value, the creation of a child process was unsuccessful */
    if (pid < 0)
    {
        UMUQFAILRETURN("spawner(", spawnerId, "): The creation of a child process was unsuccessful!");
    }
    /* otherwise, \c fork() returns a zero to the newly created child process */
    else if (pid == 0)
    {
        // If dirPath PATH is given we change to the dirPath PATH
        if (dirPath != nullptr)
        {
            if (chdir(dirPath) < 0)
            {
                UMUQFAILRETURN("Permission is denied for one of the components in the path to : \n", dirPath, "\n child process, failed to change PATH!");
            }
        }
        if (!execvp(*argv, argv))
        {
            UMUQFAILRETURN("Error occurred in child process while attempting to execute the command!");
        }
    }
    waitpid(pid, NULL, 0);
    return true;
}

bool utility::executeCommand(std::string const &inCommand, std::string const &dirPath)
{
    // If dirPath PATH is given we change to the dirPath PATH
    if (dirPath.size() > 0)
    {
        if (chdir(dirPath.c_str()) < 0)
        {
            UMUQFAILRETURN("Permission is denied for one of the components in the path to : \n", dirPath, "\n Failed to change PATH!");
        }
    }
    // Executing command
    if (!std::system(inCommand.c_str()))
    {
        UMUQWARNING("Command processor does not exists!")
    }
    return true;
}

} // namespace umuq

#endif // UMUQ_UTILITY

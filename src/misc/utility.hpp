#ifndef UMUQ_UTILITY_H
#define UMUQ_UTILITY_H

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#else
#if _XOPEN_SOURCE < 500
#define _XOPEN_SOURCE 700
#endif
#endif
#ifndef _BSD_SOURCE
#define _BSD_SOURCE 1
#endif

#include <unistd.h>    // fork, execvp, chdir
#include <sys/types.h> // waitpid
#include <sys/wait.h>  // waitpid

/*! \class utility
* \brief utility is a class which includes some helper functionality.
*	
* Utility class contains functionality for exectuing command delete a name from a filesystem or 
* unlink the files, copy file from a path to other path
* 
*/
class utility
{
  public:
    /*!
     * \brief Executing command from a spanwer
     * 
     * fork and execvp in this function are used in sequence to get a new program running as a child 
     * of a current process (spanwer with me ID).
     * 
     * \param me      Id of a spanwer
     * \param argv    A pointer to an array of character strings. More precisely, its type is char **, 
     *                which is identical to the argv array used in the main program
     * \param dir     Directory PATH in which to execute commands
     * 
     * \return true 
     * \return false 
     */
    bool executeCommand(int const me, char **argv, const char *dir = nullptr)
    {
        /*!
         * After the system call to fork()
         * Unix will make an exact copy of the parent's address space and give it to the child. 
         * Therefore, the parent and child processes have separate address spaces. 
         * 
         * fork() returns a positive value, the process ID of the child process, to the parent. 
         */
        pid_t pid = fork();

        // If fork() returns a negative value, the creation of a child process was unsuccessful
        if (pid < 0)
        {
            std::cerr << "spanwer(" << me << "):" << std::endl;
            UMUQFAILRETURN("The creation of a child process was unsuccessful");
        }
        // fork() returns a zero to the newly created child process
        else if (pid == 0)
        {
            // If dir PATH is given we change to the dir PATH
            if (dir != nullptr)
            {
                /*!
                 * The chdir() function shall cause the directory named by the pathname pointed 
                 * to by the path argument to become the current working directory
                 * 
                 * Reference:
                 * https://linux.die.net/man/3/chdir
                 */
                if (chdir(dir) < 0)
                {
                    std::cerr << "Permission is denied for one of the components in the path to : " << dir << std::endl;
                    UMUQFAILRETURN("child process, failed to change PATH!");
                }
            }

            /*!
             * The execvp() system call requires two arguments:
             * - The first argument is a character string that contains the name of a file to be executed.
             * - The second argument is a pointer to an array of character strings. More precisely, its type is char **, 
             *   which is exactly identical to the argv array used in the main program
             * 
             * When execvp() is executed, the program file given by the first argument will be loaded into the caller's 
             * address space and over-write the program there. Then, the second argument will be provided to the program 
             * and starts the execution. As a result, once the specified program file starts its execution, the original 
             * program in the caller's address space is gone and is replaced by the new program.
             * 
             * It returns a negative value if the execution fails (e.g., the request file does not exist). 
             * 
             * Reference:
             * https://linux.die.net/man/3/execvp
             */
            if (!execvp(*argv, argv))
            {
                UMUQFAILRETURN("Error occured in child process while attempting to execute the command!");
            }
        }

        /*!
         * The waitpid() system call suspends execution of the calling process until a child specified by pid argument 
         * has changed state.
         * 
         * 0 means wait for any child process whose process group ID is equal to that of the calling process
         * 
         * Reference:
         * https://linux.die.net/man/2/waitpid
         */
        waitpid(pid, NULL, 0);

        return true;
    }

    /*!
     * \brief Calls the host environment's command processor (e.g. /bin/sh, cmd.exe, command.com) with 
     * the parameter command
     * 
     * \param icommand   Input command, character string identifying the command to be run in the command 
     *                   processor. If a null pointer is given, command processor is checked for existence 
     * \param dir        Directory PATH in which to execute command
     * 
     * \return true 
     * \return false 
     */
    bool executeCommand(std::string const &icommand, std::string const &dir = std::string())
    {
        // If dir PATH is given we change to the dir PATH
        if (dir.size() > 0)
        {
            /*!
             * The chdir() function shall cause the directory named by the pathname pointed 
             * to by the path argument to become the current working directory
             * 
             * Reference:
             * https://linux.die.net/man/3/chdir
             */
            if (chdir(dir.c_str()) < 0)
            {
                std::cerr << "Permission is denied for one of the components in the path to : " << dir << std::endl;
                UMUQFAILRETURN("Failed to change PATH!");
            }
        }

        // Executing command
        if (!std::system(icommand.c_str()))
        {
            UMUQWARNING("Command processor does not exists!")
        }
        return true;
    }
};

#endif
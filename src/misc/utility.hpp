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

/*! \enum permissions
 * \brief This class represents file access permissions.
 * Access permissions model based on POSIX permission bits.
 * 
 */
enum class permissions : mode_t
{
    /*! No permission bits are set. */
    none = 0,
    /*! File owner has read permission. */
    owner_read = S_IRUSR,
    /*! File owner has write permission. */
    owner_write = S_IWUSR,
    /*! File owner has execute/search permission. */
    owner_exec = S_IXUSR,
    /*! File owner has read, write, and execute/search permissions. <br> Equivalent to `owner_read  | owner_write | owner_exec` */
    owner_all = S_IRWXU,
    /*! The file's user group has read permission. */
    group_read = S_IRGRP,
    /*! The file's user group has write permission. */
    group_write = S_IWGRP,
    /*! The file's user group has execute/search permission. */
    group_exec = S_IXGRP,
    /*! The file's user group has read, write, and execute/search permissions. <br> Equivalent to `group_read | group_write | group_exec` */
    group_all = S_IRWXG,
    /*! Other users have read permission. */
    others_read = S_IROTH,
    /*! Other users have write permission. */
    others_write = S_IWOTH,
    /*! Other users have execute/search permission. */
    others_exec = S_IXOTH,
    /*! Other users have read, write, and execute/search permissions. <br> Equivalent to `others_read | others_write | others_exec` */
    others_all = S_IRWXO,
    /*! All users have read, write, and execute/search permissions. <br> Equivalent to `owner_all | group_all | others_all` */
    all = S_IRWXU | S_IRWXG | S_IRWXO,
    /*! Set user ID to file owner user ID on execution. */
    set_uid = S_ISUID,
    /*! Set group ID to file's user group ID on execution. */
    set_gid = S_ISGID
};

/*!
 * 
 * \brief Utility mutex object
 * 
 */
static std::mutex utility_m;

/*! \class utility
 *
 * \brief This class includes some utility helpers.
 *	
 * Utility class contains functionality for exectuing commands, underneath it uses some functionalities as : <br>
 * 
 * \c fork and \c execvp are used in sequence to get a new program running as a child 
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
 * \b sync() : <br> 
 * The \c sync() system call, synchronize data on disk with memory. sync writes any data buffered in memory out to disk. 
 * This can include (but is not limited to) modified superblocks, modified inodes, and delayed reads and writes.
 * 
 * Reference: <br> 
 * https://linux.die.net/man/8/sync
 * 
 * 
 */
class utility
{
  public:
    /*!
     * \brief Get the Current Working Directory object
     * 
     * \returns std::string Current Working Directory
     */
    inline std::string getCurrentWorkingDirectory();

    /*!
     * \brief Create a new directory with name directoryName with permission mode permissionMode
     * 
     * \param directoryName   Directory Name
     * \param permissionMode  Permission mode
     * 
     * \returns false If it fails to create a directory
     */
    inline bool createDirectory(std::string const &directoryName, permissions const permissionMode = permissions::owner_all);

    /*!
     * \brief Change the process's working directory to PATH
     * 
     * \param workingDirectory   Working directory Name
     */
    inline bool changeWorkingDirectory(std::string const &workingDirectory);

    /*!
     * \brief Change the process's working directory to PATH
     * 
     * \param workingDirectory   Working directory Name
     */
    inline bool changeWorkingDirectory(char const *workingDirectory);

    /*!
     * \brief Change the process's working directory to PATH.
     * If workingDirectory does not exist first create it and then change to it.
     * 
     * \param workingDirectory   Working directory Name
     * \param permissionMode     Permission mode
     */
    bool createChangeWorkingDirectory(std::string const &workingDirectory, permissions const permissionMode = permissions::owner_all);

    /*!
     * \brief Change the process's working directory to PATH.
     * If workingDirectory does not exist first create it and then change to it.
     * 
     * \param workingDirectory   Working directory Name
     * \param permissionMode     Permission mode
     */
    bool createChangeWorkingDirectory(char const *workingDirectory, permissions const permissionMode = permissions::owner_all);

    /*!
     * \brief On successful lock acquisition of \c utility_m returns true, otherwise suspends execution of 
     * the calling thread for (at least) \c millisecondsDuration miliseconds and try again lock acquisition .
     * The sleep may be lengthened slightly by any system activity or by the time spent processing the call 
     * or by the granularity of system timers.
     *
     * \param millisecondsDuration  Sleep duration in milliseconds  
     * \param maxNumTrials          Maximum number of trials to acquire lock (per default it is set \c umuq::HugeCost). \sa umuq::HugeCost
     * 
     * \returns true   On successful lock acquisition
     * \returns false  If number of trials exceeds the maxNumTrials
     */
    bool tryLockOrSleep(int const millisecondsDuration, int const maxNumTrials = umuq::HugeCost);

    /*!
     * \brief Clone the calling process, creating an exact copy.
     * 
     * \param spawnerId  Spawner ID
     * \returns pid_t 0 to the new process, and the process ID of the new process to the old process. 
     */
    inline pid_t cloneProcess(int const spawnerId);

    /*!
     * \brief Wait for a child matching processor ID to die and make all changes done to all files actually appear on disk.
     * 
     * \param processorID  The processor ID
     */
    inline void syncProcess(pid_t const processorID);

    /*!
     * \brief Executing command from a spawner
     * 
     * Executing command from a spawner.
     * 
     * \param spawnerId               Id of a spawner
     * \param command                 Input command, character string identifying the command to be run in the command 
     *                                processor. If a null pointer is given, command processor is checked for existence 
     * \param workingDirectory        Directory PATH in which to execute the commands
     * \param createWorkingDirectory  Flag indicates whether it should create workingDirectory, when it does not exists
     * \param permissionMode          Permission mode for creating the workingDirectory. \sa umuq::permissions
     * \param millisecondsDuration    Sleep duration in milliseconds  (default is 100 milli seconds)
     * \param maxNumTrials            Maximum number of trials to acquire lock (per default it is set \c umuq::HugeCost). \sa umuq::HugeCost
     * 
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool executeCommand(int const spawnerId, std::string const &command,
                        std::string const &workingDirectory = EmptyString, bool const createWorkingDirectory = false,
                        permissions const permissionMode = permissions::owner_all,
                        int const millisecondsDuration = 100, int const maxNumTrials = umuq::HugeCost);

    /*!
     * \brief Executing command from a spawner
     * 
     * Executing command from a spawner.
     * 
     * \param spawnerId               Id of a spawner
     * \param command                 Input command, character string identifying the command to be run in the command 
     *                                processor. If a null pointer is given, command processor is checked for existence 
     * \param workingDirectory        Directory PATH in which to execute the commands
     * \param createWorkingDirectory  Flag indicates whether it should create workingDirectory, when it does not exists
     * \param permissionMode          Permission mode for creating the workingDirectory. \sa umuq::permissions
     * \param millisecondsDuration    Sleep duration in milliseconds  (default is 100 milli seconds)
     * \param maxNumTrials            Maximum number of trials to acquire lock (per default it is set \c umuq::HugeCost). \sa umuq::HugeCost
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool executeCommand(int const spawnerId, char const *command,
                        char const *workingDirectory = nullptr, bool const createWorkingDirectory = false,
                        permissions const permissionMode = permissions::owner_all,
                        int const millisecondsDuration = 100, int const maxNumTrials = umuq::HugeCost);

    /*!
     * \brief Calls the host environment's command processor (e.g. /bin/sh, cmd.exe, command.com) with 
     * the parameter command
     * 
     * \param command                 Input command, character string identifying the command to be run in the command 
     *                                processor. If a null pointer is given, command processor is checked for existence 
     * \param workingDirectory        Directory PATH in which to execute the commands
     * \param createWorkingDirectory  Flag indicates whether it should create workingDirectory, when it does not exists
     * \param permissionMode          Permission mode for creating the workingDirectory. \sa umuq::permissions
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool executeCommand(std::string const &command,
                        std::string const &workingDirectory = EmptyString, bool const createWorkingDirectory = false,
                        permissions const permissionMode = permissions::owner_all);

    /*!
     * \brief Calls the host environment's command processor (e.g. /bin/sh, cmd.exe, command.com) with 
     * the parameter command
     * 
     * \param command                 Input command, character string identifying the command to be run in the command 
     *                                processor. If a null pointer is given, command processor is checked for existence 
     * \param workingDirectory        Directory PATH in which to execute the commands
     * \param createWorkingDirectory  Flag indicates whether it should create workingDirectory, when it does not exists
     * \param permissionMode          Permission mode for creating the workingDirectory. \sa umuq::permissions
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool executeCommand(char const *command,
                        char const *workingDirectory = nullptr, bool const createWorkingDirectory = false,
                        permissions const permissionMode = permissions::owner_all);

    /*!
     * \brief Executing multiple commands from a spawner
     * 
     * Executing multiple commands from a spawner.
     * 
     * \param spawnerId               Id of a spawner
     * \param commands                Input commands, vector of character string identifying the command to be run in the command 
     *                                processor. If a null pointer is given, command processor is checked for existence 
     * \param workingDirectory        Directory PATH in which to execute the commands
     * \param createWorkingDirectory  Flag indicates whether it should create workingDirectory, when it does not exists
     * \param permissionMode          Permission mode for creating the workingDirectory. \sa umuq::permissions
     * \param millisecondsDuration    Sleep duration in milliseconds  (default is 100 milli seconds)
     * \param maxNumTrials            Maximum number of trials to acquire lock (per default it is set \c umuq::HugeCost). \sa umuq::HugeCost
     * 
     * 
     * \returns false If it encounters an unexpected problem
     */
    bool executeCommands(int const spawnerId, std::vector<std::string> const &commands,
                         std::string const &workingDirectory = EmptyString, bool const createWorkingDirectory = false,
                         permissions const permissionMode = permissions::owner_all,
                         int const millisecondsDuration = 100, int const maxNumTrials = umuq::HugeCost);
};

inline std::string utility::getCurrentWorkingDirectory()
{
    char currentWorkingDirectory[LINESIZE];
    // Get the pathname of the current working directory, and put it in LINESIZE bytes of currentWorkingDirectory.
    // It returns NULL if the directory couldn't be determined
    if (!getcwd(currentWorkingDirectory, LINESIZE))
    {
        UMUQWARNING("The pathname of the current working directory couldn't be determined!");
        UMUQFAILRETURNSTRING("");
    }
    return std::string(currentWorkingDirectory);
}

inline bool utility::createDirectory(std::string const &directoryName, permissions const permissionMode)
{
    if (mkdir(directoryName.c_str(), static_cast<std::underlying_type_t<permissions>>(permissionMode)))
    {
        UMUQFAILRETURN("Failed to create a directory [", directoryName, "] !");
    }
    return true;
}

inline bool utility::changeWorkingDirectory(std::string const &workingDirectory)
{
    if (chdir(workingDirectory.c_str()) < 0)
    {
        UMUQFAILRETURN("Permission is denied or process failed to change PATH to : \n", workingDirectory);
    }
    return true;
}

inline bool utility::changeWorkingDirectory(char const *workingDirectory)
{
    if (chdir(workingDirectory) < 0)
    {
        UMUQFAILRETURN("Permission is denied or process failed to change PATH to : \n", workingDirectory);
    }
    return true;
}

bool utility::createChangeWorkingDirectory(std::string const &workingDirectory, permissions const permissionMode)
{
    // If workingDirectory PATH is given we change to the workingDirectory PATH
    if (workingDirectory.empty())
    {
        UMUQFAILRETURN("The pathname of the working directory couldn't be determined!");
    }
    if (chdir(workingDirectory.c_str()) < 0)
    {
        // Create a new directory workingDirectory.
        if (createDirectory(workingDirectory, permissionMode))
        {
            return changeWorkingDirectory(workingDirectory);
        }
        UMUQFAILRETURN("Failed to change working path to directory [", workingDirectory, "] !");
    }
    return true;
}

bool utility::createChangeWorkingDirectory(char const *workingDirectory, permissions const permissionMode)
{
    // If workingDirectory PATH is given we change to the workingDirectory PATH
    if (!workingDirectory)
    {
        UMUQFAILRETURN("The pathname of the working directory couldn't be determined!");
    }
    if (chdir(workingDirectory) < 0)
    {
        // Create a new directory workingDirectory.
        if (createDirectory(workingDirectory, permissionMode))
        {
            return changeWorkingDirectory(workingDirectory);
        }
        UMUQFAILRETURN("Failed to change working path to directory [", workingDirectory, "] !");
    }
    return true;
}

bool utility::tryLockOrSleep(int const millisecondsDuration, int const maxNumTrials)
{
    int numTrials(0);
    while (numTrials < maxNumTrials)
    {
        // Try to lock mutex to modify 'job_shared'
        if (utility_m.try_lock())
        {
            return true;
        }

        // Stops the execution of the current thread for a specified time duration
        std::this_thread::sleep_for(std::chrono::milliseconds(millisecondsDuration));
        numTrials++;
    }
#ifdef DEBUG
    UMUQWARNING("Failed to lock the mutex after [", numTrials, "] trials!");
#endif
    return false;
}

inline pid_t utility::cloneProcess(int const spawnerId)
{
    // Clone the calling process, creating an exact copy.
    auto processorID = fork();
    // If `fork()` returns a negative value, the creation of a child process was unsuccessful
    if (processorID < 0)
    {
        UMUQFAILRETURN("In spawner(", spawnerId, "): The creation of a child process was unsuccessful!");
    }
    return processorID;
}

inline void utility::syncProcess(pid_t const processorID)
{
    // Wait for a child matching PID to die.
    waitpid(processorID, NULL, 0);
    // Make all changes done to all files actually appear on disk.
    sync();
}

bool utility::executeCommand(int const spawnerId, std::string const &command,
                             std::string const &workingDirectory, bool const createWorkingDirectory,
                             permissions const permissionMode,
                             int const millisecondsDuration, int const maxNumTrials)
{
    if (command.empty())
    {
        UMUQFAILRETURN("There is no command!");
    }

    auto currentWorkingDirectory = getCurrentWorkingDirectory();

    if (!tryLockOrSleep(millisecondsDuration, maxNumTrials))
    {
        UMUQFAILRETURN("Failed to lock the mutex!");
    }

    // Clone the calling process, creating an exact copy.
    auto processorID = cloneProcess(spawnerId);

    // otherwise, `fork()` returns a zero to the newly created child process
    if (processorID == 0)
    {
        // If workingDirectory PATH is given we change to the workingDirectory PATH
        if (!workingDirectory.empty())
        {
            if (chdir(workingDirectory.c_str()) < 0)
            {
                if (!createWorkingDirectory)
                {
                    UMUQFAILRETURN("Permission is denied from current working directory of : \n ", currentWorkingDirectory, " \n to change to : \n", workingDirectory, "\n or, child process, failed to change PATH!");
                }
                // Create a new directory workingDirectory.
                if (!createDirectory(workingDirectory, permissionMode))
                {
                    UMUQFAILRETURN("Failed to create a directory [", workingDirectory, "] !");
                }
                if (chdir(workingDirectory.c_str()) < 0)
                {
                    UMUQFAILRETURN("Permission is denied from current working directory of : \n ", currentWorkingDirectory, " \n to change to : \n", workingDirectory, "\n or, child process, failed to change PATH!");
                }
            }
        }

#ifdef DEBUG
        std::cout << "Spawner(" << spawnerId << ") : running in : \n"
                  << getCurrentWorkingDirectory() << std::endl;
#endif

        auto i = std::system(command.c_str());
    }

    utility_m.unlock();

    syncProcess(processorID);

    return true;
}

bool utility::executeCommand(int const spawnerId, char const *command,
                             char const *workingDirectory, bool const createWorkingDirectory,
                             permissions const permissionMode,
                             int const millisecondsDuration, int const maxNumTrials)
{
    if (!command)
    {
        UMUQFAILRETURN("There is no command!");
    }

    auto currentWorkingDirectory = getCurrentWorkingDirectory();

    if (!tryLockOrSleep(millisecondsDuration, maxNumTrials))
    {
        UMUQFAILRETURN("Failed to lock the mutex!");
    }

    // Clone the calling process, creating an exact copy.
    auto processorID = cloneProcess(spawnerId);

    // otherwise, `fork()` returns a zero to the newly created child process
    if (processorID == 0)
    {
        // If workingDirectory PATH is given we change to the workingDirectory PATH
        if (workingDirectory)
        {
            if (chdir(workingDirectory) < 0)
            {
                if (!createWorkingDirectory)
                {
                    UMUQFAILRETURN("Permission is denied from current working directory of : \n ", currentWorkingDirectory, " \n to change to : \n", workingDirectory, "\n or, child process, failed to change PATH!");
                }
                // Create a new directory workingDirectory.
                if (!createDirectory(workingDirectory, permissionMode))
                {
                    UMUQFAILRETURN("Failed to create a directory [", workingDirectory, "] !");
                }
                if (chdir(workingDirectory) < 0)
                {
                    UMUQFAILRETURN("Permission is denied from current working directory of : \n ", currentWorkingDirectory, " \n to change to : \n", workingDirectory, "\n or, child process, failed to change PATH!");
                }
            }
        }

#ifdef DEBUG
        std::cout << "Spawner(" << spawnerId << ") : running in : \n"
                  << getCurrentWorkingDirectory() << std::endl;
#endif

        auto i = std::system(command);
    }

    utility_m.unlock();

    syncProcess(processorID);

    return true;
}

bool utility::executeCommand(std::string const &command,
                             std::string const &workingDirectory, bool const createWorkingDirectory,
                             permissions const permissionMode)
{
    if (command.empty())
    {
        UMUQFAILRETURN("There is no command!");
    }

    auto currentWorkingDirectory = getCurrentWorkingDirectory();

    // If workingDirectory PATH is given we change to the workingDirectory PATH
    if (!workingDirectory.empty())
    {
        if (chdir(workingDirectory.c_str()) < 0)
        {
            if (!createWorkingDirectory)
            {
                UMUQFAILRETURN("Permission is denied from current working directory of : \n ", currentWorkingDirectory, " \n to change to : \n", workingDirectory, "\n or, child process, failed to change PATH!");
            }
            // Create a new directory workingDirectory.
            if (!createDirectory(workingDirectory, permissionMode))
            {
                UMUQFAILRETURN("Failed to create a directory [", workingDirectory, "] !");
            }
            if (chdir(workingDirectory.c_str()) < 0)
            {
                UMUQFAILRETURN("Permission is denied from current working directory of : \n ", currentWorkingDirectory, " \n to change to : \n", workingDirectory, "\n or, child process, failed to change PATH!");
            }
        }
    }

#ifdef DEBUG
    std::cout << "Running in : \n"
              << getCurrentWorkingDirectory() << std::endl;
#endif

    auto i = std::system(command.c_str());

    // If we changed to the workingDirectory PATH, we need to return back.
    return workingDirectory.empty() ? true : changeWorkingDirectory(currentWorkingDirectory);
}

bool utility::executeCommand(char const *command,
                             char const *workingDirectory, bool const createWorkingDirectory,
                             permissions const permissionMode)
{
    if (!command)
    {
        UMUQFAILRETURN("There is no command!");
    }

    auto currentWorkingDirectory = getCurrentWorkingDirectory();

    // If workingDirectory PATH is given we change to the workingDirectory PATH
    if (workingDirectory)
    {
        if (chdir(workingDirectory) < 0)
        {
            if (!createWorkingDirectory)
            {
                UMUQFAILRETURN("Permission is denied from current working directory of : \n ", currentWorkingDirectory, " \n to change to : \n", workingDirectory, "\n or, child process, failed to change PATH!");
            }
            // Create a new directory workingDirectory.
            if (!createDirectory(workingDirectory, permissionMode))
            {
                UMUQFAILRETURN("Failed to create a directory [", workingDirectory, "] !");
            }
            if (chdir(workingDirectory) < 0)
            {
                UMUQFAILRETURN("Permission is denied from current working directory of : \n ", currentWorkingDirectory, " \n to change to : \n", workingDirectory, "\n or, child process, failed to change PATH!");
            }
        }
    }

#ifdef DEBUG
    std::cout << "Running in : \n"
              << getCurrentWorkingDirectory() << std::endl;
#endif

    auto i = std::system(command);

    // If we changed to the workingDirectory PATH, we need to return back.
    return workingDirectory ? true : changeWorkingDirectory(currentWorkingDirectory);
}

bool utility::executeCommands(int const spawnerId, std::vector<std::string> const &commands,
                              std::string const &workingDirectory, bool const createWorkingDirectory,
                              permissions const permissionMode,
                              int const millisecondsDuration, int const maxNumTrials)
{
    if (commands.size() == 0)
    {
        UMUQFAILRETURN("There is no command!");
    }

    auto currentWorkingDirectory = getCurrentWorkingDirectory();

    if (!tryLockOrSleep(millisecondsDuration, maxNumTrials))
    {
        UMUQFAILRETURN("Failed to lock the mutex!");
    }

    // Clone the calling process, creating an exact copy.
    auto processorID = cloneProcess(spawnerId);

    // otherwise, `fork()` returns a zero to the newly created child process
    if (processorID == 0)
    {
        // If workingDirectory PATH is given we change to the workingDirectory PATH
        if (!workingDirectory.empty())
        {
            if (chdir(workingDirectory.c_str()) < 0)
            {
                if (!createWorkingDirectory)
                {
                    UMUQFAILRETURN("Permission is denied from current working directory of : \n ", currentWorkingDirectory, " \n to change to : \n", workingDirectory, "\n or, child process, failed to change PATH!");
                }
                // Create a new directory workingDirectory.
                if (!createDirectory(workingDirectory, permissionMode))
                {
                    UMUQFAILRETURN("Failed to create a directory [", workingDirectory, "] !");
                }
                if (chdir(workingDirectory.c_str()) < 0)
                {
                    UMUQFAILRETURN("Permission is denied from current working directory of : \n ", currentWorkingDirectory, " \n to change to : \n", workingDirectory, "\n or, child process, failed to change PATH!");
                }
            }
        }

#ifdef DEBUG
        std::cout << "Spawner(" << spawnerId << ") : running in : \n"
                  << getCurrentWorkingDirectory() << std::endl;
#endif

        for (auto c = 0; c < commands.size(); c++)
        {
            auto i = std::system(commands[c].c_str());
        }
    }

    utility_m.unlock();

    syncProcess(processorID);

    return true;
}

} // namespace umuq

#endif // UMUQ_UTILITY

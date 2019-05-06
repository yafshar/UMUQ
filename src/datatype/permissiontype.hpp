#ifndef UMUQ_PERMISSIONTYPE_H
#define UMUQ_PERMISSIONTYPE_H

#include <sys/stat.h>

namespace umuq
{

/*! \enum permissionType
 * \brief This class represents file access permission.
 * 
 * File access permission is the access permissions model based on POSIX permission bits.
 * 
 */
enum class permissionType : mode_t
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

} // namespace umuq

#endif // UMUQ_PERMISSIONTYPE

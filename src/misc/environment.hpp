#ifndef UMUQ_ENVIRONMENT_H
#define UMUQ_ENVIRONMENT_H

#ifdef HAVE_CONFIG_H
#include <UMUQ_config.h>
#endif /* HAVE_CONFIG_H */

#ifdef HAVE_LIBTORC
#ifdef HAVE_LIBGTEST
#include "gtest/gtest.h"

/*!
 * \brief torcEnvironment class
 * 
 */
class torcEnvironment : public ::testing::Environment
#else
class torcEnvironment
#endif
{
  public:
    /*!
     * \brief Set up the torc object
     * 
     */
    virtual void SetUp()
    {
        char **argv;
        int argc = 0;

        torc_init(argc, argv, 0);
    }

    /*!
     * \brief Set up the torc object
     * 
     * \param argc 
     * \param argv 
     */
    virtual void SetUp(int argc, char **argv)
    {
        torc_init(argc, argv, 0);
    }

    /*!
     * \brief Finish the torc
     * 
     */
    virtual void TearDown()
    {
        torc_finalize();
    }

    /*!
     * \brief Destroy the torc Environment object
     * 
     */
    virtual ~torcEnvironment() {}
};
#else
/*!
 * \brief torcEnvironment class
 * 
 */
class torcEnvironment
{
  public:
    /*!
     * \brief Set up the torc object
     * 
     */
    virtual void SetUp() {}
   
    /*!
     * \brief Set up the torc object
     * 
     * \param argc 
     * \param argv 
     */
    virtual void SetUp(int argc, char **argv) {}

    /*!
     * \brief Finish the torc
     * 
     */
    virtual void TearDown() {}

    /*!
     * \brief Destroy the torc Environment object
     * 
     */
    virtual ~torcEnvironment() {}
};
#endif //HAVE_LIBTORC
#endif //UMUQ_ENVIRONMENT_H

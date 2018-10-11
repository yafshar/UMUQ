#ifndef UMUQ_TIMER_H
#define UMUQ_TIMER_H

#include <iostream>
#include <sys/time.h>

namespace umuq
{

/*! \class UMUQTimer
 * 
 * \brief Start stopwatch timer class
 *
 * - \b tic starts a stopwatch timer, and stores the internal time at execution of the command.
 * - \b toc displays the elapsed time so that you can record time for simultaneous time spans.
 * 
 * Consecutive tic overwrites the previous recorded time.
 */
class UMUQTimer
{
  public:
    /*!
     * \brief Construct a new UMUQTimer object
     * 
     */
    inline UMUQTimer();

    /*!
     * \brief Destroy the UMUQTimer object
     * 
     */
    ~UMUQTimer();

    /*!
    * \brief tic starts a stopwatch timer, and stores the internal time at execution of the command. 
    *
    *  tic starts a stopwatch timer, and stores the internal time at execution of the command. 
    *  Consecutive tic overwrites the previous recorded time.
    */
    inline void tic();

    /*!
    * \brief toc displays the elapsed time so that you can record time for simultaneous time spans.
    *
    *  toc displays the elapsed time so that you can record time for simultaneous time spans. 
    */
    inline void toc();
    inline void toc(std::string const &timing_name);

  private:
    inline double my_gettime();

  private:
    double t1_internal;
    double t2_internal;
};

inline UMUQTimer::UMUQTimer() { tic(); }

UMUQTimer::~UMUQTimer() {}

inline void UMUQTimer::tic()
{
    t1_internal = my_gettime();
}

inline void UMUQTimer::toc()
{
    t2_internal = my_gettime();

    double elapsed_seconds = t2_internal - t1_internal;

    // output the elapsed time to terminal
    std::cout << " It took " << std::to_string(elapsed_seconds) << " seconds" << std::endl;
}

inline void UMUQTimer::toc(std::string const &timing_name)
{
    t2_internal = my_gettime();

    double elapsed_seconds = t2_internal - t1_internal;

    // output the elapsed time to terminal
    std::cout << timing_name << " took " << std::to_string(elapsed_seconds) << " seconds" << std::endl;
}

inline double UMUQTimer::my_gettime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1.0E-6;
}

} // namespace umuq

#endif // UMUQ_TIMER

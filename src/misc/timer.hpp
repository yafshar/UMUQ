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
 * tic starts a stopwatch timer, and stores the internal time at execution of the command.
 * 
 * toc displays the elapsed time so that you can record time for simultaneous time spans.
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
    UMUQTimer();

    /*!
     * \brief Destroy the UMUQTimer object
     * 
     */
    ~UMUQTimer();

    /*!
    *  \brief tic starts a stopwatch timer, and stores the internal time at execution of the command. 
    *
    *  tic starts a stopwatch timer, and stores the internal time at execution of the command. 
    *  Consecutive tic overwrites the previous recorded time.
    */
    void tic();

    /*!
    *  \brief toc displays the elapsed time so that you can record time for simultaneous time spans.
    *
    *  toc displays the elapsed time so that you can record time for simultaneous time spans. 
    */
    void toc();
    void toc(std::string const &timing_name);

  private:
    double my_gettime();

  private:
    double t1_internal;
    double t2_internal;
};

UMUQTimer::UMUQTimer()
{
    tic();
}

UMUQTimer::~UMUQTimer() {}

void UMUQTimer::tic()
{
    t1_internal = my_gettime();
}

void UMUQTimer::toc()
{
    t2_internal = my_gettime();

    double elapsed_seconds = t2_internal - t1_internal;

    //output the elapsed time to terminal
    std::cout << " It took " << std::to_string(elapsed_seconds) << " seconds" << std::endl;
}

void UMUQTimer::toc(std::string const &timing_name)
{
    t2_internal = my_gettime();

    double elapsed_seconds = t2_internal - t1_internal;

    //output the elapsed time to terminal
    std::cout << timing_name << " took " << std::to_string(elapsed_seconds) << " seconds" << std::endl;
}

double UMUQTimer::my_gettime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1.0E-6;
}

} // namespace umuq

#endif // UMUQ_TIMER

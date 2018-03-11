#ifndef UMHBM_TIMER_H
#define UMHBM_TIMER_H

#include <iostream>
#include <sys/time.h>

/*! \brief Start stopwatch timer class
*
* tic starts a stopwatch timer, and stores the internal time at execution of the command.
* 
* toc displays the elapsed time so that you can record time for simultaneous time spans.
* 
* Consecutive tic overwrites the previous recorded time.
*/
class UMTimer
{
  public:
    UMTimer() {
        tic();
    }

    void tic()
    {
        t1_internal = my_gettime();
    }

    void toc()
    {
        t2_internal = my_gettime();

        double elapsed_seconds = t2_internal - t1_internal;

        //output the elapsed time to terminal 
        std::cout << " It took " << std::to_string(elapsed_seconds) << " seconds" << std::endl;
    }

    void toc(std::string timing_name)
    {
        t2_internal = my_gettime();

        double elapsed_seconds = t2_internal - t1_internal;

        //output the elapsed time to terminal 
        std::cout << timing_name << " took " << std::to_string(elapsed_seconds) << " seconds" << std::endl;
    }

  private:
    double t1_internal;
    double t2_internal;

    double my_gettime()
    {
        struct timeval t;
        gettimeofday(&t, NULL);
        return (double)t.tv_sec + (double)t.tv_usec * 1.0E-6;
    }
};

#endif
#ifndef UMUQ_TIMER_H
#define UMUQ_TIMER_H

#include <iostream>
#include <sys/time.h>

namespace umuq
{

/*! \class umuqTimer
 * 
 * \brief Start stopwatch timer class
 *
 * - \b tic starts a stopwatch timer, and stores the internal time at execution of the command.
 * - \b toc displays the elapsed time so that you can record time for simultaneous time spans.
 * 
 * \note
 * - Consecutive tic overwrites the previous recorded time.
 */
class umuqTimer
{
  public:
    /*!
     * \brief Construct a new umuqTimer object
     * 
     * \param CoutFlag Flag indicator whether it should print output to a stream buffer (default is true)
     */
    umuqTimer(bool const CoutFlag = true);

    /*!
     * \brief Destroy the umuqTimer object
     * 
     */
    ~umuqTimer();

    /*!
    * \brief tic starts a stopwatch timer, and stores the internal time at execution of the command. 
    *
    * It starts a stopwatch timer, and stores the internal time at execution of the command. 
    * Consecutive tic overwrites the previous recorded time.
    */
    inline void tic();

    /*!
    * \brief toc displays the elapsed time so that you can record time for simultaneous time spans.
    *
    * It displays the elapsed time so that you can record time for simultaneous time spans. 
    */
    inline void toc();

    /*!
    * \brief toc displays the elapsed time so that you can record time for simultaneous time spans.
    *
    * It displays the elapsed time so that you can record time for simultaneous time spans. 
    */
    inline void toc(std::string const &functionName);

    /*!
     * \brief It would print the measured elapsed interval times and corresponding function names
     * 
     */
    inline void print();

  public:
    /*! Indicator flag whether we should print output to a stream buffer or not */
    bool coutFlag;

    /*! If \c coutFlag is false, it would keep the measured elapsed interval times */
    std::vector<double> timeInetrval;

    /*! If \c coutFlag is false, it would keep the name of the function for each measrued interval */
    std::vector<std::string> timeInetrvalFunctionNames;

  private:
    /*! The first time point. Time point 1 */
    std::chrono::system_clock::time_point timePoint1;

    /*! The second time point. Time point 2 */
    std::chrono::system_clock::time_point timePoint2;

    /*! Counter for the cases where we do not pass function names */
    std::size_t callCounter;
};

umuqTimer::umuqTimer(bool const CoutFlag) : coutFlag(CoutFlag), callCounter(0) { tic(); }

umuqTimer::~umuqTimer() {}

inline void umuqTimer::tic() { timePoint1 = std::chrono::system_clock::now(); }

inline void umuqTimer::toc()
{
    timePoint2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedTime = timePoint2 - timePoint1;
    if (coutFlag)
    {
        std::cout << " It took " << std::to_string(elapsedTime.count()) << " seconds" << std::endl;
        return;
    }
    timeInetrval.push_back(elapsedTime.count());
    timeInetrvalFunctionNames.push_back(std::to_string(callCounter++));
}

inline void umuqTimer::toc(std::string const &functionName)
{
    timePoint2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedTime = timePoint2 - timePoint1;
    if (coutFlag)
    {
        std::cout << functionName << " took " << std::to_string(elapsedTime.count()) << " seconds" << std::endl;
        return;
    }
    timeInetrval.push_back(elapsedTime.count());
    timeInetrvalFunctionNames.push_back(functionName);
}

inline void umuqTimer::print()
{
    auto functionIt = timeInetrvalFunctionNames.begin();
    for (auto timerIt = timeInetrval.begin(); timerIt != timeInetrval.end(); timerIt++, functionIt++)
    {
        std::cout << *functionIt << " took " << std::to_string(*timerIt) << " seconds" << std::endl;
    }
}

} // namespace umuq

#endif // UMUQ_TIMER

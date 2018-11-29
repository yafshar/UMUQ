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
 * Consecutive tic overwrites the previous recorded time.
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
     * \brief If \c coutFlag is false, it would print the measured elapsed interval times and corresponding function names
     * 
     */
    void print();

  public:
    /*!
     * \brief Indicator flag whether we should print output to a stream buffer or not
     * 
     */
    bool coutFlag;

  public:
    /*!
     * \brief If \c coutFlag is false, it would keep the measured elapsed interval times
     * 
     */
    std::vector<double> timeInetrval;

    /*!
     * \brief If \c coutFlag is false, it would keep the name of the function for each measrued interval
     * 
     */
    std::vector<std::string> timeInetrvalFunctionNames;

  private:
    /*!
     * \brief Time point 1 
     * 
     */
    std::chrono::system_clock::time_point timePoint1;

    /*!
     * \brief Time point 2
     * 
     */
    std::chrono::system_clock::time_point timePoint2;
};

umuqTimer::umuqTimer(bool const CoutFlag) : coutFlag(CoutFlag) { tic(); }

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

void umuqTimer::print()
{
    if (!coutFlag)
    {
        if (timeInetrvalFunctionNames.size() > 0)
        {
            auto functionIt = timeInetrvalFunctionNames.begin();
            for (auto timerIt = timeInetrval.begin(); timerIt != timeInetrval.end(); timerIt++, functionIt++)
            {
                std::cout << *functionIt << " took " << std::to_string(*timerIt) << " seconds" << std::endl;
            }
        }
        else
        {
            int Counter(0);
            for (auto timerIt = timeInetrval.begin(); timerIt != timeInetrval.end(); timerIt++)
            {
                std::cout << Counter++ << " took " << std::to_string(*timerIt) << " seconds" << std::endl;
            }
        }
    }
}

} // namespace umuq

#endif // UMUQ_TIMER

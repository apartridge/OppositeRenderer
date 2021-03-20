/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "BenchmarkTimer.h"

BenchmarkTimer::BenchmarkTimer(void)
    : m_state(NOT_RUNNING),
      m_accumSeconds(0.0f)
{

}

BenchmarkTimer::~BenchmarkTimer( void )
{

}

void BenchmarkTimer::start()
{
    m_time.start();
    m_accumSeconds = 0;
    m_state = RUNNING;
}

void BenchmarkTimer::restart()
{
    m_time.restart();
    m_accumSeconds = 0.0f;
    m_state = RUNNING;
}

void BenchmarkTimer::pause()
{
    if(m_state == RUNNING)
    {
        m_accumSeconds += m_time.elapsed()/1000.0f;
    }
    m_state = NOT_RUNNING;
}

void BenchmarkTimer::resume()
{
    if(m_state == NOT_RUNNING)
    {
        m_time.restart();
        m_state = RUNNING;
    }
}

double BenchmarkTimer::elapsedSeconds()
{
    BenchmarkTimerState state = m_state;

    if(state == RUNNING)
    {
        pause();
    }

    double secs = m_accumSeconds;
    
    if(state == RUNNING)
    {
        resume();
    }

    return secs;
}

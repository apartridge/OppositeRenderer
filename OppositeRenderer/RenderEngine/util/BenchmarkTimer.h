/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include <QTime>
#include "render_engine_export_api.h"

class BenchmarkTimer
{
public:
    RENDER_ENGINE_EXPORT_API BenchmarkTimer(void);
    RENDER_ENGINE_EXPORT_API ~BenchmarkTimer(void);
    RENDER_ENGINE_EXPORT_API void start();
    RENDER_ENGINE_EXPORT_API void restart();
    RENDER_ENGINE_EXPORT_API void pause();
    RENDER_ENGINE_EXPORT_API void resume();
    RENDER_ENGINE_EXPORT_API double elapsedSeconds();
private:
    enum BenchmarkTimerState 
    {
        RUNNING, NOT_RUNNING
    };
    QTime m_time;
    double m_accumSeconds;
    BenchmarkTimerState m_state; 
};


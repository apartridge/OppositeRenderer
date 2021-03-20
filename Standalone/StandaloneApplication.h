/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#pragma once

#include <memory>

#include "Application.h"
#include "StandaloneRenderManager.h"

class QApplication;
class ComputeDevice;
class QThread;

class StandaloneApplication : public Application
{
public:
    StandaloneApplication(QApplication& qApplication, const ComputeDevice& device);
    void wait();

private:
    StandaloneRenderManager m_renderManager;
    std::unique_ptr<QThread> m_thread;
};
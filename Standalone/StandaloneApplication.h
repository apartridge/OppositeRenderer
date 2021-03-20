/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

#include "Application.hxx"
#include "StandaloneRenderManager.hxx"

class QApplication;
class ComputeDevice;
class QThread;
class StandaloneApplication : public Application
{
public:
    StandaloneApplication(QApplication & qApplication, const ComputeDevice & device);
    ~StandaloneApplication(void);
    void wait();
private:
    StandaloneRenderManager m_renderManager;
    QThread* m_thread;
};
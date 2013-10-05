/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "StandaloneApplication.h"
#include "ComputeDevice.h"
#include <QApplication>
#include <QThread>

StandaloneApplication::StandaloneApplication(QApplication & qApplication, const ComputeDevice & device)
    : Application(qApplication),
      m_renderManager(StandaloneRenderManager(qApplication, *this, device))
{
    connect(&m_renderManager, SIGNAL(newFrameReadyForDisplay(const float*, unsigned long long)), 
            this, SIGNAL(newFrameReadyForDisplay(const float*, unsigned long long)));

    // Run render manager in thread

    m_thread = new QThread(&qApplication);
    m_renderManager.moveToThread(m_thread);
    QObject::connect(m_thread, SIGNAL(started()), &m_renderManager, SLOT(start()));
    m_thread->start();

    // Pass on render manager errors as application errors
    connect(&m_renderManager, SIGNAL(renderManagerError(QString)), 
        this, SIGNAL(applicationError(QString)),
        Qt::QueuedConnection);

}

StandaloneApplication::~StandaloneApplication(void)
{

}

void StandaloneApplication::wait()
{
    waitOnApplicationFinished();
    m_thread->quit();
    m_thread->wait();
}

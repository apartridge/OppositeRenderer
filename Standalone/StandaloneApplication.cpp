/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "StandaloneApplication.h"
#include "ComputeDevice.h"

#include <QApplication>
#include <QThread>

StandaloneApplication::StandaloneApplication(QApplication& qApplication, const ComputeDevice& device)
    : Application(qApplication)
    , m_renderManager(qApplication, *this, device)
{
    connect(
        &m_renderManager,
        &StandaloneRenderManager::newFrameReadyForDisplay,
        this,
        &StandaloneApplication::newFrameReadyForDisplay);

    // Run render manager in thread
    m_thread = std::make_unique<QThread>(&qApplication);
    m_renderManager.moveToThread(m_thread.get());
    QObject::connect(m_thread.get(), &QThread::started, &m_renderManager, &StandaloneRenderManager::start);
    m_thread->start();

    // Pass on render manager errors as application errors
    connect(
        &m_renderManager,
        &StandaloneRenderManager::renderManagerError,
        this,
        &Application::applicationError,
        Qt::QueuedConnection);
}

void StandaloneApplication::wait()
{
    waitOnApplicationFinished();
    m_thread->quit();
    m_thread->wait();
}

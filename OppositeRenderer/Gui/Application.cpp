/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "Application.hxx"
#include "RunningStatus.h"
#include <QMetaType>
#include <QApplication>
#include <QThread>
#include <QMessageBox>

Application::Application(QApplication & qApplication) :
    m_sequenceNumber(0),
    m_runningStatus(RunningStatus::STOPPED),
    m_renderMethod(RenderMethod::PROGRESSIVE_PHOTON_MAPPING),
    m_rendererStatus(RendererStatus::NOT_INITIALIZED)
{
    qRegisterMetaType<RunningStatus::E>("RunningStatus::E");

    // Move scene manager to a thread
    m_sceneManagerThread = new QThread(&qApplication);
    m_sceneManager.moveToThread(m_sceneManagerThread);
    m_sceneManagerThread->start();

    connect(&m_sceneManager, SIGNAL(sceneUpdated()), this, SLOT(onSceneUpdated()));
    connect(&m_sceneManager, SIGNAL(sceneLoadingNew()), this, SLOT(onSceneLoadingNew()));
    connect(&m_sceneManager, SIGNAL(sceneLoadError(QString)), this, SLOT(onSceneLoadError(QString)));
    
    m_outputSettingsModel.setWidth(1280);
    m_outputSettingsModel.setHeight(720);
    m_outputSettingsModel.setGamma(2.2f);
    m_PPMSettingsModel.setPPMInitialRadius(0.20);
    m_sceneManager.setScene("Cornell");

    connect(&m_outputSettingsModel, SIGNAL(resolutionUpdated()), this, SLOT(onOutputSettingsUpdated()));
    connect(&m_PPMSettingsModel, SIGNAL(updated()), this, SLOT(onPPMSettingsUpdated()));

    onOutputSettingsUpdated();
    onPPMSettingsUpdated();
}

void Application::onAboutToQuit()
{
    setRunningStatus(RunningStatus::STOPPED);
}

void Application::waitOnApplicationFinished()
{
    m_sceneManagerThread->quit();
    m_sceneManagerThread->wait();
}

OutputSettingsModel & Application::getOutputSettingsModel()
{
    return m_outputSettingsModel;
}

PPMSettingsModel & Application::getPPMSettingsModel()
{
    return m_PPMSettingsModel;
}

Camera & Application::getCamera()
{
    return m_camera;
}

SceneManager & Application::getSceneManager()
{
    return m_sceneManager;
}

const SceneManager & Application::getSceneManager() const
{
    return m_sceneManager;
}

RenderMethod::E Application::getRenderMethod() const
{
    return m_renderMethod;
}

void Application::setRenderMethod(RenderMethod::E method)
{
    incrementSequenceNumber();
    m_renderMethod = method;
    emit renderMethodChanged();
}

unsigned int Application::getWidth() const
{
    return m_outputSettingsModel.getWidth();
}

unsigned int Application::getHeight() const
{
    return m_outputSettingsModel.getHeight();
}

void Application::onCameraUpdated()
{
    emit cameraUpdated();
    incrementSequenceNumber();
}

unsigned long long Application::getSequenceNumber() const
{
    return m_sequenceNumber;
}

void Application::incrementSequenceNumber()
{
    m_sequenceNumber++;
    m_renderStatisticsModel.setNumIterations(0);
    m_renderStatisticsModel.setNumPreviewedIterations(0);
    resetRenderTime();
    emit sequenceNumberIncremented();
}

void Application::onRenderRestart()
{
    incrementSequenceNumber();
    setRunningStatus(RunningStatus::RUNNING);
}

void Application::onRenderStatusToggle()
{
    if(m_runningStatus == RunningStatus::STOPPED)
    {
        onRenderRestart();
    }
    else if(m_runningStatus == RunningStatus::RUNNING)
    {
        setRunningStatus(RunningStatus::PAUSE);
        pauseRenderTime();
    }
    else
    {
        setRunningStatus(RunningStatus::RUNNING);
        resumeRenderTime();
    }
}

void Application::setCameraToSceneDefault()
{
    m_camera = m_sceneManager.getScene()->getDefaultCamera();
    m_camera.setAspectRatio(m_outputSettingsModel.getAspectRatio());
    onCameraUpdated();
}

void Application::onOutputSettingsUpdated()
{
    m_camera.setAspectRatio(m_outputSettingsModel.getAspectRatio());
    onCameraUpdated();
}

void Application::onPPMSettingsUpdated()
{
    incrementSequenceNumber();
}

RenderStatisticsModel & Application::getRenderStatisticsModel()
{
    return m_renderStatisticsModel;
}

const RenderStatisticsModel & Application::getRenderStatisticsModel() const
{
    return m_renderStatisticsModel;
}

float Application::getRenderTimeSeconds() const
{
    if(m_runningStatus == RunningStatus::RUNNING)
    {
        return m_renderElapsedSeconds + ((float)m_renderTime.elapsed()/1000);
    }
    return m_renderElapsedSeconds;
}

void Application::resetRenderTime()
{
    m_renderTime.restart();
    m_renderElapsedSeconds = 0;
}

void Application::pauseRenderTime()
{
    m_renderElapsedSeconds += m_renderTime.elapsed()/1000.0f;
}

void Application::resumeRenderTime()
{
    m_renderTime.restart();
}

void Application::onSceneLoadingNew()
{
    setRunningStatus(RunningStatus::PAUSE);
}

void Application::onSceneUpdated()
{
    m_camera = m_sceneManager.getScene()->getDefaultCamera();
    m_camera.setAspectRatio(m_outputSettingsModel.getAspectRatio());
    m_PPMSettingsModel.setPPMInitialRadius(m_sceneManager.getScene()->getSceneInitialPPMRadiusEstimate());
    onCameraUpdated();
    onRenderRestart();
}

void Application::onSceneLoadError(QString error)
{
    emit applicationError(error);
}

void Application::onNewFrameReadyForDisplay(const float*, unsigned long long )
{
    m_renderStatisticsModel.incrementNumPreviewedIterations();
}

RendererStatus::E Application::getRendererStatus() const
{
    return m_rendererStatus;
}

void Application::setRendererStatus( RendererStatus::E val )
{
    bool shouldEmit = m_rendererStatus != val;
    m_rendererStatus = val;
    if(shouldEmit)
    {
        emit rendererStatusChanged();
    }
}

RunningStatus::E Application::getRunningStatus() const
{
    return m_runningStatus;
}

void Application::setRunningStatus( RunningStatus::E val )
{
    bool shouldEmit = m_runningStatus != val;
    m_runningStatus = val;
    if(shouldEmit)
    {
        emit runningStatusChanged();
    }
}

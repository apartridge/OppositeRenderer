/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "StandaloneRenderManager.h"
#include "Application.hxx"
#include "clientserver/RenderServerRenderRequest.h"
#include "renderer/Camera.h"
#include "renderer/OptixRenderer.h"
#include "scene/Scene.h"

#include <QApplication>
#include <QCoreApplication>
#include <QThread>
#include <QTime>

StandaloneRenderManager::StandaloneRenderManager(
    QApplication& qApplication, Application& application, const ComputeDevice& device)
    : m_device(device)
    , m_nextIterationNumber(0)
    , m_currentScene(nullptr)
    , m_compileScene(false)
    , m_application(application)
    , m_noEmittedSignals(true)
{
    connect(&application,
            &Application::sequenceNumberIncremented,
            this,
            &StandaloneRenderManager::onSequenceNumberIncremented);

    connect(&application, &Application::runningStatusChanged, this, &StandaloneRenderManager::onRunningStatusChanged);

    connect(&application.getSceneManager(),
            &SceneManager::sceneLoadingNew,
            this,
            &StandaloneRenderManager::onSceneLoadingNew);

    connect(&application.getSceneManager(), &SceneManager::sceneUpdated,
            this,
            &StandaloneRenderManager::onSceneUpdated);

    onSceneUpdated();

    connect(this, &StandaloneRenderManager::continueRayTracing, this, &StandaloneRenderManager::onContinueRayTracing, Qt::QueuedConnection);
}

void StandaloneRenderManager::start()
{
    m_application.setRendererStatus(RendererStatus::INITIALIZING_ENGINE);
    m_renderer.initialize(m_device);
}

void StandaloneRenderManager::onContinueRayTracing()
{
    renderNextIteration();
    continueRayTracingIfRunningAsync();
}

void StandaloneRenderManager::renderNextIteration()
{
    try
    {
        if (m_application.getRunningStatus() == RunningStatus::RUNNING && m_currentScene != NULL)
        {
            m_noEmittedSignals = true;

            if (m_compileScene)
            {
                m_application.setRendererStatus(RendererStatus::INITIALIZING_SCENE);
                m_renderer.initScene(*m_currentScene);
                m_compileScene = false;
                m_application.setRendererStatus(RendererStatus::RENDERING);
            }

            // We only display one every X frames on screen (to make fair comparison with distributed renderer)
            bool shouldOutputIteration = m_nextIterationNumber % 5 == 0;

            const double PPMAlpha = 2.0 / 3.0;
            QVector<unsigned long long> iterationNumbers;
            QVector<double> ppmRadii;

            RenderServerRenderRequestDetails details(
                m_camera,
                QByteArray(m_currentScene->getSceneName()),
                m_application.getRenderMethod(),
                m_application.getWidth(),
                m_application.getHeight(),
                PPMAlpha);

            RenderServerRenderRequest renderRequest(
                m_application.getSequenceNumber(), iterationNumbers, ppmRadii, details);

            m_renderer.renderNextIteration(
                m_nextIterationNumber,
                m_nextIterationNumber,
                m_PPMRadius,
                shouldOutputIteration,
                renderRequest.getDetails());
            const double ppmRadiusSquared = m_PPMRadius * m_PPMRadius;
            const double ppmRadiusSquaredNew
                = ppmRadiusSquared * (m_nextIterationNumber + PPMAlpha) / double(m_nextIterationNumber + 1);
            m_PPMRadius = sqrt(ppmRadiusSquaredNew);

            // Transfer the output buffer to CPU and signal ready for display

            if (shouldOutputIteration)
            {
                if (!m_outputBuffer)
                {
                    m_outputBuffer = std::make_unique<float[]>(2000 * 2000 * 3);
                }
                m_renderer.getOutputBuffer(m_outputBuffer.get());
                emit newFrameReadyForDisplay(m_outputBuffer.get(), m_nextIterationNumber);
            }

            fillRenderStatistics();
            m_nextIterationNumber++;
        }
    }
    catch (const std::exception& E)
    {
        m_application.setRunningStatus(RunningStatus::PAUSE);
        QString error = QString("%1").arg(E.what());
        emit renderManagerError(error);
    }
}

void StandaloneRenderManager::fillRenderStatistics()
{
    m_application.getRenderStatisticsModel().setNumIterations(m_nextIterationNumber);
    m_application.getRenderStatisticsModel().setCurrentPPMRadius(m_PPMRadius);

    if (m_application.getRenderMethod() == RenderMethod::PROGRESSIVE_PHOTON_MAPPING)
    {
        m_application.getRenderStatisticsModel().setNumEmittedPhotonsPerIteration(
            OptixRenderer::EMITTED_PHOTONS_PER_ITERATION);
        m_application.getRenderStatisticsModel().setNumEmittedPhotons(
            OptixRenderer::EMITTED_PHOTONS_PER_ITERATION * m_nextIterationNumber);
    }
    else
    {
        m_application.getRenderStatisticsModel().setNumEmittedPhotonsPerIteration(0);
        m_application.getRenderStatisticsModel().setNumEmittedPhotons(0);
    }
}

// TODO this may be called very often for rapid camera changes.
void StandaloneRenderManager::onSequenceNumberIncremented()
{
    m_nextIterationNumber = 0;
    m_PPMRadius = m_application.getPPMSettingsModel().getPPMInitialRadius();
    m_camera = m_application.getCamera();
    continueRayTracingIfRunningAsync();
}

void StandaloneRenderManager::onSceneLoadingNew()
{
}

void StandaloneRenderManager::onSceneUpdated()
{
    IScene* scene = m_application.getSceneManager().getScene();
    if (scene != m_currentScene)
    {
        m_compileScene = true;
        m_currentScene = scene;
    }
}

void StandaloneRenderManager::continueRayTracingIfRunningAsync()
{
    if (m_application.getRunningStatus() == RunningStatus::RUNNING && m_noEmittedSignals)
    {
        m_noEmittedSignals = false;
        emit continueRayTracing();
    }
}

void StandaloneRenderManager::onRunningStatusChanged()
{
    continueRayTracingIfRunningAsync();
}
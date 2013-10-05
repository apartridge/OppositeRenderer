/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "RenderServerRenderer.hxx"
#include "scene/Cornell.h"
#include "clientserver/RenderServerRenderRequest.h"
#include "clientserver/RenderResultPacket.h"
#include <QTime>
#include <QMetaType>
#include "RenderServer.hxx"
#include <QCoreApplication>
#include <QThread>
#include <QByteArray>

RenderServerRenderer::RenderServerRenderer(const RenderServer & renderServer) :
    m_renderServer(renderServer),
    m_renderer(OptixRenderer()),
    m_computeDevice(NULL),
    m_scene(NULL),
    m_quit(false),
    m_currentSequenceNumber(0)
{
    qRegisterMetaType<RenderResultPacket>("RenderResultPacket");
    qRegisterMetaType<RenderServerRenderRequest>("RenderServerRenderRequest");
}

RenderServerRenderer::~RenderServerRenderer(void)
{
    delete m_scene;
    m_scene = NULL;
}

void RenderServerRenderer::onAboutToQuit()
{
    m_quit = true;
}

void RenderServerRenderer::wait()
{
    m_waitCondition.wakeAll();
    m_currentSequenceNumber++;
}

void RenderServerRenderer::onThreadStarted()
{
    onNewRenderCommandInQueue();
}

void RenderServerRenderer::initialize(const ComputeDevice* computeDevice)
{
    m_renderer.initialize(*computeDevice);
    m_computeDevice = computeDevice;
    m_totalTime.start();
    m_renderTime.start();
    m_currentSequenceNumber = 0;
    m_queue.clear();
}

void RenderServerRenderer::initializeNewClient()
{
    m_queueMutex.lock();
    m_currentSequenceNumber = 0;
    m_queue.clear();
    m_queueMutex.unlock();
}

void RenderServerRenderer::onNewRenderCommandInQueue()
{
    while(true)
    {
        // Process queued up events/slots, in case we quit
        QCoreApplication::processEvents();

        if(m_quit)
        {
            return;
        }

        // If queue is empty, we wait until we get a new request
        if(m_queue.size() == 0)
        {
            m_waitCondition.wait(&m_waitConditionMutex);
            continue;
        }
        // Process the next RenderServerRenderRequest

        RenderServerRenderRequest renderRequest = m_queue.dequeue();
        QString iterationNumbersInPacketString = "";

        for(int i = 0; i < renderRequest.getNumIterations(); i++)
        {            
            // If the packet we are working has become old during this rendering for-loop, then break out of this loop
            if(m_currentSequenceNumber != renderRequest.getSequenceNumber())
            {
                break;
            }
            // Else we'll render this frame
            else
            {
                // This renderRequest has a new scene name, so we'll load the new scene
                if(m_scene == NULL || m_scene->getSceneName() != renderRequest.getDetails().getSceneName())
                {
                    loadNewScene(renderRequest.getDetails().getSceneName());
                }

                // Render the frame with local iteration number going from 0 to renderRequestsCurrentPacket.size()
                // We only need to create the output buffer for the last iteration of the packet
                bool createOutputBuffer = i == renderRequest.getNumIterations() - 1;
                renderFrame(renderRequest.getIterationNumbers().at(i), i, renderRequest.getPPMRadii().at(i), createOutputBuffer, renderRequest.getDetails());
                iterationNumbersInPacketString += " " + QString::number(renderRequest.getIterationNumbers().at(i));
            }
        }

        // If the packet we have just rendered is old (m_currentSequenceNumber is newer)
        // then we drop this packet. This can happen if we have started on a RenderServerRenderRequest but later found out about
        // a new sequence, in which case we have break-ed out of the loop above.

        if(renderRequest.getSequenceNumber() == m_currentSequenceNumber)
        {
            RenderResultPacket result = createRenderResultPacket(renderRequest);
            QString logString = QString("TRANSFERRING packet (%1 iteration:%2) in sequence %3 to client.")
                .arg(result.getNumIterationsInPacket())
                .arg(iterationNumbersInPacketString)
                .arg(result.getSequenceNumber());
            emit newLogString(logString);
            emit newRenderResultPacket(result);
        }
        else
        {
            QString logString = QString("IGNORED package with %1 iterations since sequence %3 != %4.")
                .arg(renderRequest.getNumIterations())
                .arg(renderRequest.getSequenceNumber())
                .arg(m_currentSequenceNumber);
            emit newLogString(logString);
        }
    }
}

void RenderServerRenderer::renderFrame(unsigned long long iterationNumber, unsigned long long localIterationNumber, 
    float PPMRadius, bool createOutputBuffer, const RenderServerRenderRequestDetails & details)
{
    // We perform the rendering using m_renderer

    BenchmarkTimer frameTime;
    frameTime.start();
    m_renderTime.resume();
    m_renderer.renderNextIteration(iterationNumber, localIterationNumber, PPMRadius, createOutputBuffer, details);
    m_renderTime.pause();
    double frameRenderTime = frameTime.elapsedSeconds();

    QString logString = QString("RENDERED iteration # %1 in %6 s. (%3x%4 PPM-r: %5)")
        .arg(iterationNumber)
        .arg(details.getWidth())
        .arg(details.getHeight())
        .arg(PPMRadius)
        .arg(frameRenderTime, 0, 'g', 2);
    emit newLogString(logString);
}

RenderResultPacket RenderServerRenderer::createRenderResultPacket(const RenderServerRenderRequest & request)
{
    QByteArray outputBuffer;
    int bufferSizeBytes = m_renderer.getScreenBufferSizeBytes();
    outputBuffer.resize(bufferSizeBytes);
    m_renderer.getOutputBuffer(outputBuffer.data());
    RenderResultPacket result = RenderResultPacket(request.getSequenceNumber(), request.getIterationNumbers(), outputBuffer);
    return result;
}

const ComputeDevice & RenderServerRenderer::getComputeDevice() const
{
    return *m_computeDevice;
}

void RenderServerRenderer::pushCommandToQueue( RenderServerRenderRequest renderRequest )
{
    m_queueMutex.lock();
    if(renderRequest.getSequenceNumber() > m_currentSequenceNumber)
    {
        m_currentSequenceNumber = renderRequest.getSequenceNumber();
        m_renderTime.restart();
        m_totalTime.restart();
    }
    m_queue.enqueue(renderRequest);
    m_queueMutex.unlock();
    m_waitCondition.wakeAll();
}

void RenderServerRenderer::onClientDisconnected()
{
    m_queueMutex.lock();
    m_currentSequenceNumber = 0;
    m_queueMutex.unlock();
}

void RenderServerRenderer::loadNewScene(const QByteArray & sceneNameB )
{
    try
    {
        emit newLogString(QString("INITIALIZING scene %1. Please wait...").arg(QString(sceneNameB)));

        delete m_scene;
        m_scene = NULL;

        SceneFactory factory;
        const char* sceneName = sceneNameB.constData();
        m_scene = factory.getSceneByName(sceneName);
        m_renderer.initScene(*m_scene);

        emit newLogString(QString("INITIALIZED scene %1.").arg(QString(m_scene->getSceneName())));
    }
    catch(const std::exception & E)
    {
        emit newLogString(QString("An error happened during loading of scene %1: \"%2\"")
            .arg(QString(sceneNameB)).arg(QString(E.what())));
    }
    catch(...)
    {
        emit newLogString(QString("An unknown error happened during loading of scene: %s.")
            .arg(QString(sceneNameB)));
    }
}

unsigned int RenderServerRenderer::getNumPendingRenderCommands()
{
    m_queueMutex.lock();
    int sum = m_queue.size();
    m_queueMutex.unlock();
    return sum;
}

double RenderServerRenderer::getRenderTimeSeconds()
{
    return m_renderTime.elapsedSeconds();
}

double RenderServerRenderer::getTotalTimeSeconds()
{
    return m_totalTime.elapsedSeconds();
}

unsigned long long RenderServerRenderer::getCurrentSequenceNumber() const
{
    return m_currentSequenceNumber;
}

unsigned int RenderServerRenderer::getNumPendingRenderIterations()
{
    m_queueMutex.lock();
    unsigned int iterations = 0;
    for(int i = 0; i < m_queue.size(); i++)
    {
        iterations += m_queue.at(i).getNumIterations();
    }
    m_queueMutex.unlock();
    return iterations;
}

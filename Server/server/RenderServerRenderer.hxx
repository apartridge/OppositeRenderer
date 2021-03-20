/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "renderer/OptixRenderer.h"
#include "clientserver/RenderResultPacket.h"
#include "clientserver/RenderServerRenderRequest.h"
#include "scene/SceneFactory.h"
#include <QString>
#include <QObject>
#include <QMutex>
#include <QQueue>
#include <QWaitCondition>
#include "util/BenchmarkTimer.h"

class ComputeDevice;
class RenderServer;

class RenderServerRenderer : public QObject
{
    Q_OBJECT;
public:
    RenderServerRenderer(const RenderServer & renderServer);
    ~RenderServerRenderer(void);
    void initialize(const ComputeDevice* computeDevice);
    void initializeNewClient();
    const ComputeDevice & getComputeDevice() const;
    void pushCommandToQueue( RenderServerRenderRequest renderRequest );
    unsigned int getNumPendingRenderCommands();
    unsigned long long getCurrentSequenceNumber() const;
    double getRenderTimeSeconds();
    double getTotalTimeSeconds();
    void wait();
    unsigned int getNumPendingRenderIterations();

public slots:
    void onThreadStarted();
    void onAboutToQuit();
    void onClientDisconnected();

signals:
    void newLogString(QString);
    void newRenderResultPacket(RenderResultPacket);

private slots:
    void onNewRenderCommandInQueue();

private:
    void renderFrame(unsigned long long iterationNumber, unsigned long long localIterationNumber, float PPMRadius, bool createOutputBuffer, const RenderServerRenderRequestDetails & details);
    RenderResultPacket createRenderResultPacket(const RenderServerRenderRequest & request);
    void loadNewScene(const QByteArray & sceneName  );
    const RenderServer & m_renderServer;
    OptixRenderer m_renderer;
    const ComputeDevice* m_computeDevice;
    QString m_sceneName;
    IScene* m_scene;

    BenchmarkTimer m_totalTime;
    BenchmarkTimer m_renderTime;
    
    unsigned long long m_currentSequenceNumber;

    QMutex m_queueMutex;
    QQueue<RenderServerRenderRequest> m_queue;
    QWaitCondition m_waitCondition;
    QMutex m_waitConditionMutex;
    bool m_quit;
};


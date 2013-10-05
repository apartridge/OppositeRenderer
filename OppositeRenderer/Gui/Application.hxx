/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

#include <QObject>
#include "RunningStatus.h"
#include "renderer/RenderMethod.h"
#include "renderer/Camera.h"
#include "gui_export_api.h"
#include "models/OutputSettingsModel.hxx"
#include "models/PPMSettingsModel.hxx"
#include "models/RenderStatisticsModel.hxx"
#include "scene/SceneManager.hxx"
#include "RendererStatus.h"

class IScene;
class QApplication;
class QThread;

/*
A base class interface for Applications, which are the API between the window and Applications
to communicate. A note is that the set functions should be "asynchronous" in the implementations, i.e. they
do not directly modify the state but emit a signal which is then caught and executed by the Application
in its own thread.
*/

class GUI_EXPORT_API_QT Application : public QObject
{
    Q_OBJECT;
public:

    Application(QApplication & qApplication);
    ~Application(){};

    OutputSettingsModel & getOutputSettingsModel();
    unsigned int getWidth() const ;
    unsigned int getHeight() const ;
    PPMSettingsModel & getPPMSettingsModel();
    Camera & getCamera();
    RenderStatisticsModel & getRenderStatisticsModel();
    const RenderStatisticsModel & getRenderStatisticsModel() const;
    void setCameraToSceneDefault();
    SceneManager & getSceneManager();
    const SceneManager & getSceneManager() const;
    void setRenderMethod( RenderMethod::E method );
    RenderMethod::E getRenderMethod() const;
    RunningStatus::E getRunningStatus() const;
    void setRunningStatus(RunningStatus::E val);

    RendererStatus::E getRendererStatus() const;
    void setRendererStatus(RendererStatus::E val);

    unsigned long long getSequenceNumber() const;
    float getRenderTimeSeconds() const;
    void waitOnApplicationFinished();

public slots:
    void onRenderRestart();
    void onRenderStatusToggle();

private slots:
    void onCameraUpdated();
    void onAboutToQuit();
    void onOutputSettingsUpdated();
    void onPPMSettingsUpdated();
    void onSceneLoadingNew();
    void onSceneUpdated();
    void onSceneLoadError(QString);
    void onNewFrameReadyForDisplay(const float*, unsigned long long);

signals:
    void runningStatusChanged();
    void rendererStatusChanged();
    void renderMethodChanged();
    void cameraUpdated();
    void newFrameReadyForDisplay(const float*, unsigned long long);
    void sequenceNumberIncremented();
    void applicationError(QString);

private:
    QThread* m_sceneManagerThread;
    RunningStatus::E m_runningStatus;
    RendererStatus::E m_rendererStatus;
    RenderMethod::E m_renderMethod;
    OutputSettingsModel m_outputSettingsModel;
    PPMSettingsModel m_PPMSettingsModel;
    RenderStatisticsModel m_renderStatisticsModel;
    SceneManager m_sceneManager;
    Camera m_camera;
    Camera m_defaultCamera;
    unsigned long long m_sequenceNumber;
    void incrementSequenceNumber();
    void resetRenderTime();
    void pauseRenderTime();
    void resumeRenderTime();
    QTime m_renderTime;
    float m_renderElapsedSeconds;
};

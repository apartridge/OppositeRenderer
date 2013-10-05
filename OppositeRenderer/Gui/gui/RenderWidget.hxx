/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

#include <QObject>
#include <QGLWidget>
#include <QPair>
#include "util/Mouse.h"
#include "renderer/Camera.h"
#include <QTime>

class OptixRenderer;
class RenderWindow;
class QThread;
class ComputeDevice;
class OutputSettingsModel;
class QLabel;

/*
 * Handles displaying the Application output to the screen.
*/

class RenderWidget : public QGLWidget
{
    Q_OBJECT

public:
    RenderWidget(QWidget *parent, Camera & camera, const OutputSettingsModel & model);
    ~RenderWidget();
    size_t getDisplayBufferSizeBytes();

signals:
    void cameraUpdated();

public slots:
    void onNewFrameReadyForDisplay(const float* cpuBuffer, unsigned long long iterationNumber);

protected:
    virtual void initializeGL();
    virtual void resizeGL(int w, int h);
    virtual void paintGL();
    void displayFrame(const float* cpuBuffer, unsigned long long iterationNumber);
    QPair<int, int> getDisplayBufferSize();
    virtual void mousePressEvent(QMouseEvent* event);
    virtual void mouseMoveEvent( QMouseEvent* event );
    virtual void resizeEvent(QResizeEvent* event);
        
private:
    void initializeOpenGLShaders();
    Mouse m_mouse;
    float* m_displayBufferCpu;
    Camera & m_camera;
    const OutputSettingsModel & m_outputSettingsModel;
    bool m_hasLoadedGLShaders;
    GLuint m_GLProgram;
    GLuint m_GLTextureSampler;
    GLuint m_GLOutputBufferTexture;
    QLabel* m_iterationNumberLabel;
};
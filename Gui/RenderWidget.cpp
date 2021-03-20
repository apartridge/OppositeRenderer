/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

/*
Handles the OpenGL context.
Handles outputting the image to the screen including gamma correction using a OpenGL shader
Takes mouse events and delegate them to the Mouse class
*/

#include "RenderWidget.h"
#include "models/OutputSettingsModel.h"
#include "renderer/Camera.h"

#include <QLabel>
#include <QMessageBox>
#include <QMouseEvent>
#include <QThread>

RenderWidget::RenderWidget(QWidget* parent, Camera& camera, const OutputSettingsModel& outputSettings)
    : QOpenGLWidget(parent)
    , m_camera(camera)
    , m_outputSettingsModel(outputSettings)
    , m_hasLoadedGLShaders(false)
    , m_GLProgram(0)
    , m_GLTextureSampler(0)
    , m_currentIterationNumber(0)
{
    resize(outputSettings.getWidth(), outputSettings.getHeight());
    setMouseTracking(false);

    m_displayBufferCpu = std::make_unique<float[]>(2000 * 2000 * 3);

    m_iterationNumberLabel = new QLabel(this);
    m_iterationNumberLabel->setStyleSheet("background:rgb(51,51,51); font-size:20pt; color:rgb(170,170,170);");
    m_iterationNumberLabel->setAlignment(Qt::AlignRight);
    m_iterationNumberLabel->hide();
}

void RenderWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0.2f, 0.2f, 0.2f, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    if (!m_hasLoadedGLShaders)
    {
        initializeOpenGLShaders();
    }
}

void RenderWidget::onNewFrameReadyForDisplay(const float* cpuBuffer, unsigned long long iterationNumber)
{
    assert(cpuBuffer);
    memcpy(m_displayBufferCpu.get(), cpuBuffer, getDisplayBufferSizeBytes());
    m_currentIterationNumber = iterationNumber;
    update();
}

void RenderWidget::resizeGL(int w, int h)
{
    update();
}

void RenderWidget::paintGL()
{
    if (!m_displayBufferCpu)
    {
        return;
    }
    glClear(GL_COLOR_BUFFER_BIT);

    int offsetX = ((int)size().width() - (int)m_outputSettingsModel.getWidth()) / 2;
    int offsetY = ((int)size().height() - (int)m_outputSettingsModel.getHeight()) / 2;

    if (offsetY > 20)
    {
        m_iterationNumberLabel->show();
        m_iterationNumberLabel->setText(QString::number(m_currentIterationNumber));
        m_iterationNumberLabel->setGeometry(
            offsetX + m_outputSettingsModel.getWidth() - 250, offsetY + m_outputSettingsModel.getHeight() + 5, 250, 30);
    }
    else
    {
        m_iterationNumberLabel->hide();
    }

    glViewport(offsetX, offsetY, (GLint)m_outputSettingsModel.getWidth(), (GLint)m_outputSettingsModel.getHeight());

    glBindTexture(GL_TEXTURE_2D, m_GLOutputBufferTexture);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB32F,
        m_outputSettingsModel.getWidth(),
        m_outputSettingsModel.getHeight(),
        0,
        GL_RGB,
        GL_FLOAT,
        (GLvoid*)m_displayBufferCpu.get());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLint loc = glGetUniformLocation(m_GLProgram, "invgamma");
    if (loc != -1)
    {
        glUniform1f(loc, 1.0 / m_outputSettingsModel.getGamma());
    }

    glEnable(GL_TEXTURE_2D);

    glBegin(GL_QUADS);

    glTexCoord2f(0, 0);
    glVertex2f(0, 0);

    glTexCoord2f(1, 0);
    glVertex2f(1, 0);

    glTexCoord2f(1, 1);
    glVertex2f(1, 1);

    glTexCoord2f(0, 1);
    glVertex2f(0, 1);

    glEnd();

    glDisable(GL_TEXTURE_2D);
}

size_t RenderWidget::getDisplayBufferSizeBytes()
{
    int width = m_outputSettingsModel.getWidth();
    int height = m_outputSettingsModel.getHeight();
    return width * height * 3 * sizeof(float);
}

void RenderWidget::mousePressEvent(QMouseEvent* event)
{
}

void RenderWidget::mouseMoveEvent(QMouseEvent* event)
{
}

void RenderWidget::initializeOpenGLShaders()
{
    m_hasLoadedGLShaders = true;

    const char* shaderSource = "uniform sampler2D sceneBuffer; "
                               "uniform float invgamma;"
                               "void main(){ "
                               "    vec2 uv = gl_TexCoord[0].xy;"
                               "    vec3 color = texture2D(sceneBuffer, uv).rgb;"
                               "    gl_FragColor.rgb = pow(color, vec3(invgamma));"
                               "    gl_FragColor.a = 1.0;"
                               "}";

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(fragmentShader, 1, &shaderSource, NULL);

    glCompileShader(fragmentShader);

    GLint validCompilation = 0;
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &validCompilation);

    if (!validCompilation)
    {
        GLint blen = 0;
        GLsizei slen = 0;

        glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &blen);
        if (blen > 1)
        {
            GLchar compiler_log[200];
            glGetShaderInfoLog(fragmentShader, sizeof(compiler_log) - 1, &slen, compiler_log);
            printf("compiler_log:%s\n", compiler_log);
            QMessageBox::warning(this, "Error compiling OpenGL shader!", QString(compiler_log));
            exit(0);
        }
    }

    m_GLProgram = glCreateProgram();
    glAttachShader(m_GLProgram, fragmentShader);
    glLinkProgram(m_GLProgram);
    glUseProgram(m_GLProgram);

    glGenSamplers(1, &m_GLTextureSampler);
    glSamplerParameteri(m_GLTextureSampler, GL_TEXTURE_WRAP_S, GL_REPEAT);

    glGenTextures(1, &m_GLOutputBufferTexture);
    glBindSampler(m_GLOutputBufferTexture, m_GLTextureSampler);
}
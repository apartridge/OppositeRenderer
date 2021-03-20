/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "SceneManager.hxx"
#include <QMetaObject>
#include <QString>

SceneManager::SceneManager(void)
    : m_scene(NULL)
    , m_status(SceneManagerStatus::NO_SCENE)
{
}

SceneManager::~SceneManager(void)
{
    delete m_scene;
}

IScene* SceneManager::getScene() const
{
    return m_scene;
}

// Asynchronously load the new scene given by sceneName
void SceneManager::setScene(const char* sceneName)
{
    QMetaObject::invokeMethod(this, "onLoadNewScene", Qt::QueuedConnection, Q_ARG(QString, QString(sceneName)));
}

void SceneManager::onLoadNewScene(QString sceneName)
{
    emit sceneLoadingNew();
    SceneManagerStatus::E oldStatus = m_status;
    m_status = SceneManagerStatus::IMPORTING;
    try
    {
        IScene* oldScene = m_scene;
        m_scene = m_factory.getSceneByName(sceneName.toLatin1().constData());
        emit sceneUpdated();
        delete oldScene;
        m_status = SceneManagerStatus::HAS_SCENE;
    }
    catch (const std::exception& E)
    {
        emit sceneLoadError(QString(E.what()));
        m_status = oldStatus;
    }
}

SceneManagerStatus::E SceneManager::getStatus() const
{
    return m_status;
}

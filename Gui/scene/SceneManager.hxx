/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#pragma once

#include "../gui_export_api.h"
#include "SceneFactory.h"
#include "scene/IScene.h"

#include <QObject>

enum class SceneManagerStatus
{
    NO_SCENE,
    IMPORTING,
    HAS_SCENE
};

class SceneManager : public QObject
{
    Q_OBJECT;

public:
    SceneManager();
    GUI_EXPORT_API_QT IScene* getScene() const;
    GUI_EXPORT_API_QT void setScene(const char* sceneName);
    SceneManagerStatus getStatus() const;
signals:
    void sceneLoadingNew();
    void sceneUpdated();
    void sceneLoadError(QString error);
private slots:
    void onLoadNewScene(QString sceneName);

private:
    SceneFactory m_factory;
    SceneManagerStatus m_status;
    std::unique_ptr<IScene> m_scene;
};
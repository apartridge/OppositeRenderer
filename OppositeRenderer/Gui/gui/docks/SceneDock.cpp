/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "SceneDock.hxx"
#include "ui/ui_SceneDock.h"
#include <QMessageBox>
#include "scene/SceneManager.hxx"

SceneDock::SceneDock(QWidget *parent, SceneManager & sceneManager) :
    QDockWidget(parent),
    m_sceneManager(sceneManager),
    ui(new Ui::SceneDock)
{
    ui->setupUi(this);
    this->setFeatures(QDockWidget::DockWidgetFloatable|QDockWidget::DockWidgetMovable);
    this->setAllowedAreas(Qt::LeftDockWidgetArea|Qt::RightDockWidgetArea);
    connect(&sceneManager, SIGNAL(sceneUpdated()), this, SLOT(onSceneUpdated()));
    connect(&sceneManager, SIGNAL(sceneLoadingNew()), this, SLOT(onSceneLoadingNew()));
    connect(&sceneManager, SIGNAL(sceneLoadError(QString)), this, SLOT(onSceneUpdated()));
    onSceneUpdated();
}

SceneDock::~SceneDock()
{
    delete ui;
}

void SceneDock::onSceneUpdated()
{
    if(m_sceneManager.getScene() != NULL)
    {
        ui->sceneNameLabel->setText(QString(m_sceneManager.getScene()->getSceneName()));
        ui->numTrianglesLabel->setText(QString::number(m_sceneManager.getScene()->getNumTriangles()));
    }
}

void SceneDock::onSceneLoadingNew()
{
    ui->sceneNameLabel->setText(QString("Loading new scene..."));
    ui->numTrianglesLabel->setText(QString(""));
}

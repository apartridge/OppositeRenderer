/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "RenderInformationDock.hxx"
#include "ui/ui_RenderInformationDock.h"
#include "Application.hxx"
#include "models/RenderStatisticsModel.hxx"
#include <Qtimer>
#include "gui/MainWindowBase.hxx"

RenderInformationDock::RenderInformationDock(QWidget *parent, const RenderStatisticsModel & renderStatisticsModel, Application & application) :
    QDockWidget(parent),
    ui(new Ui::RenderInformationDock),
    m_renderStatisticsModel(renderStatisticsModel),
    m_application(application)
{
    ui->setupUi(this);
    this->setMinimumSize(QSize(242, 38));
    this->setFeatures(QDockWidget::DockWidgetFloatable|QDockWidget::DockWidgetMovable);
    this->setAllowedAreas(Qt::LeftDockWidgetArea|Qt::RightDockWidgetArea);

    connect(&application, SIGNAL(runningStatusChanged()), 
        this, SLOT(onRunningStatusChanged()));

    connect(&renderStatisticsModel, SIGNAL(updated()), 
        this, SLOT(onRenderStatisticsUpdated()));

    connect(ui->buttonRestartRender, SIGNAL(clicked()), this, SIGNAL(renderRestart()));
    connect(ui->buttonStatusToggleRender, SIGNAL(clicked()), this, SIGNAL(renderStatusToggle()));

    QTimer* timer = new QTimer(this);
    timer->setInterval(200);
    connect(timer, SIGNAL(timeout()), this, SLOT(onUpdateRenderTime()));
    timer->start();
   
    onRunningStatusChanged();
    onUpdateRenderTime();
}

RenderInformationDock::~RenderInformationDock()
{
    delete ui;
}

void RenderInformationDock::onRunningStatusChanged()
{
    if(m_application.getRunningStatus() == RunningStatus::PAUSE)
    {
        ui->buttonRestartRender->setText("Restart render");
        ui->buttonRestartRender->setDisabled(false);
        ui->buttonStatusToggleRender->setText("Resume render");
        ui->buttonStatusToggleRender->setDisabled(false);
    }
    else if(m_application.getRunningStatus() == RunningStatus::STOPPED)
    {
        ui->buttonStatusToggleRender->setText("Resume Render");
        ui->buttonStatusToggleRender->setDisabled(true);
        ui->buttonRestartRender->setDisabled(false);
        ui->buttonRestartRender->setText("Start render");
    }
    else
    {
        ui->buttonRestartRender->setText("Restart render");
        ui->buttonRestartRender->setDisabled(false);
        ui->buttonStatusToggleRender->setText("Pause render");
        ui->buttonStatusToggleRender->setDisabled(false);
    }
}

void RenderInformationDock::onRenderStatisticsUpdated()
{
    float elapsed = m_application.getRenderTimeSeconds();
    unsigned long long iterationNumber = m_renderStatisticsModel.getNumIterations();
    ui->iterationNumberLabel->setText(QString::number(iterationNumber));
    onUpdateRenderTime();
}

void RenderInformationDock::onUpdateRenderTime()
{
    ui->runningStatusLabel->setText(MainWindowBase::getApplicationStatusString(m_application, false));
    ui->renderTimeLabel->setText(QString("%1 seconds").arg(m_application.getRenderTimeSeconds(), 0, 'f', 1));
    
    if(m_application.getRenderTimeSeconds() > 0.5 && m_renderStatisticsModel.getNumIterations() > 0)
    {
        float iterationsPerSecond = m_renderStatisticsModel.getNumIterations()/m_application.getRenderTimeSeconds();
        ui->iterationsPerSecondLabel->setText(QString("%1").arg(iterationsPerSecond, 0, 'f', 4));
    }
    else
    {
        ui->iterationsPerSecondLabel->setText("");
    }

}

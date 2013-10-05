/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "PPMDock.hxx"
#include "ui/ui_PPMDock.h"
#include <QMessageBox>
#include "Application.hxx"
#include "models/PPMSettingsModel.hxx"

PPMDock::PPMDock(QWidget *parent, const Application & application, PPMSettingsModel & PPMSettingsModel) :
    QDockWidget(parent),
    ui(new Ui::PPMDock),
    m_application(application),
    m_PPMSettingsModel(PPMSettingsModel)
{
    ui->setupUi(this);
    this->setFeatures(QDockWidget::DockWidgetFloatable|QDockWidget::DockWidgetMovable);
    this->setAllowedAreas(Qt::LeftDockWidgetArea|Qt::RightDockWidgetArea);
    connect(ui->updateSettingsButton, SIGNAL(pressed()), this, SLOT(onFormSubmitted()));
    connect(&PPMSettingsModel, SIGNAL(updated()), this, SLOT(onModelUpdated()));
    connect(&m_application.getRenderStatisticsModel(), SIGNAL(updated()), this, SLOT(onRenderStatisticsUpdated()));
    onModelUpdated();
}

PPMDock::~PPMDock()
{
    delete ui;
}

void PPMDock::onFormSubmitted()
{
    m_PPMSettingsModel.setPPMInitialRadius(ui->ppmInitialRadiusEdit->value());
}

void PPMDock::onRenderStatisticsUpdated()
{
    const RenderStatisticsModel & model = m_application.getRenderStatisticsModel();
    ui->emittedPhotons->setText(QString("%1 M").arg(model.getNumEmittedPhotons()/1E6, 0, 'f', 2));
    ui->ppmCurrentRadius->setText(QString("%1").arg(model.getCurrentPPMRadius(), 0, 'f', 3));
    ui->emittedPhotonsPerIteration->setText(QString("%1 M").arg(model.getNumEmittedPhotonsPerIteration()/1E6, 0, 'f', 2));
    if(m_application.getRenderTimeSeconds() > 0.5f)
    {
        ui->emittedPhotonsPerSecond->setText(QString("%1 M").arg(model.getNumEmittedPhotons()/m_application.getRenderTimeSeconds()/1E6, 0, 'f', 2));
    }
    else
    {
        ui->emittedPhotonsPerSecond->setText("");
    }
}

void PPMDock::onModelUpdated()
{
    ui->ppmInitialRadiusEdit->setValue(m_PPMSettingsModel.getPPMInitialRadius());
}

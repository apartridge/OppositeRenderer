/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "OutputDock.hxx"
#include "ui/ui_OutputDock.h"
#include "models/OutputSettingsModel.hxx"
#include <QMessageBox>

OutputDock::OutputDock(QWidget *parent, OutputSettingsModel & model) :
    QDockWidget(parent),
    ui(new Ui::OutputDock),
    m_model(model)
{
    ui->setupUi(this);
    this->setFeatures(QDockWidget::DockWidgetFloatable|QDockWidget::DockWidgetMovable);
    this->setAllowedAreas(Qt::LeftDockWidgetArea|Qt::RightDockWidgetArea);
    connect(ui->updateSettingsButton, SIGNAL(pressed()), this, SLOT(onFormSubmitted()));
    connect(&m_model, SIGNAL(resolutionUpdated()), this, SLOT(onOutputSettingsModelUpdated()));
    connect(&m_model, SIGNAL(gammaUpdated()), this, SLOT(onOutputSettingsModelUpdated()));


    onOutputSettingsModelUpdated();
}

OutputDock::~OutputDock()
{
    delete ui;
}

void OutputDock::onFormSubmitted()
{
    bool okWidth, okHeight;
    unsigned int width = ui->resolutionWidthEdit->text().toUInt(&okWidth);
    unsigned int height = ui->resolutionHeightEdit->text().toUInt(&okHeight);
    float gamma = (float)ui->gammaEdit->value();

    if(okWidth && okHeight && width < 10000 && height < 10000)
    {
        m_model.setWidth(width);
        m_model.setHeight(height);
        m_model.setGamma(gamma);
    }
    else
    {
        QMessageBox::information(this, "Invalid Render Output Settings", "Please make sure that the values you inserted are valid.");
    }
}

void OutputDock::onOutputSettingsModelUpdated()
{
    ui->resolutionWidthEdit->setText(QString::number(m_model.getWidth()));
    ui->resolutionHeightEdit->setText(QString::number(m_model.getHeight()));
    ui->gammaEdit->setValue((double)m_model.getGamma());
}

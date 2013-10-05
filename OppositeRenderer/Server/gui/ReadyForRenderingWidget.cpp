/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "ReadyForRenderingWidget.hxx"
#include "ui/ui_ReadyForRenderingWidget.h"
#include "ComputeDevice.h"

ReadyForRenderingWidget::ReadyForRenderingWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ReadyForRenderingWidget)
{
    ui->setupUi(this);
}

ReadyForRenderingWidget::~ReadyForRenderingWidget()
{
    delete ui;
}

void ReadyForRenderingWidget::appendToLog(const QString& log)
{
    ui->serverLog->appendPlainText(log);
}

void ReadyForRenderingWidget::clearLog()
{
    ui->serverLog->clear();
}

void ReadyForRenderingWidget::setClientName(const QString& name)
{
    ui->clientLabel->setText(name);
}

void ReadyForRenderingWidget::setComputeDevice(const ComputeDevice & device )
{
    ui->computeDeviceNameLabel->setText(QString("%1 (id %2)").arg(device.getName()).arg(device.getDeviceId()));
}

void ReadyForRenderingWidget::setServerName( const QString & name )
{
    ui->serverLabel->setText(name);
}

void ReadyForRenderingWidget::setRenderTime( float renderTime, float totalTime )
{
    ui->totalTimeLabel->setText(QString("%1 seconds").arg(totalTime, 0, 'f', 1));
    ui->timeSpentRenderingLabel->setText(QString("%1 seconds (%2 %)").arg(renderTime).arg((renderTime/totalTime)*100, 0, 'f', 1));
}

void ReadyForRenderingWidget::setPendingRenderCommandsAndIterations( unsigned int pendingRenderIterations, unsigned int pendingRenderCommands  )
{
   ui->pendingRenderIterationsCommandsLabel->setText(QString("%1/%2").arg(pendingRenderIterations).arg(pendingRenderCommands));
}
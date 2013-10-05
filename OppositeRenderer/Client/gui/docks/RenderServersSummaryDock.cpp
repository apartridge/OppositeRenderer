#include "RenderServersSummaryDock.hxx"
#include "ui/ui_RenderServersSummaryDock.h"
#include "client/RenderServerConnections.hxx"
#include "DistributedApplication.hxx"

RenderServersSummaryDock::RenderServersSummaryDock(QWidget *parent, DistributedApplication & application) :
    QDockWidget(parent),
    ui(new Ui::RenderServersSummaryDock),
    m_application(application)
{
    ui->setupUi(this);
    connect(&m_application.getServerConnections(), SIGNAL(serversStateUpdated()), this, SLOT(onServersInfoUpdated()));
    connect(ui->pushButtonNewServerConnection, SIGNAL(clicked()), this, SIGNAL(actionConnectToNewRenderServer()));
}

RenderServersSummaryDock::~RenderServersSummaryDock()
{
    delete ui;
}

void RenderServersSummaryDock::onServersInfoUpdated()
{
    ui->previewedIterationsLabel->setText(QString::number(m_application.getRenderStatisticsModel().getNumPreviewedIterations()));
    ui->packetsPendingLabel->setText(QString::number(m_application.getTotalPacketsPending()));
    ui->numIterationsInBufferLabel->setText(QString::number(m_application.getBackBufferNumIterations()));
    ui->backBufferSizeLabel->setText(QString("%1 MB (peak %2 MB)")
                        .arg(float(m_application.getBackBufferSizeBytes())/1024/1024, 0, 'f', 1)
                        .arg(float(m_application.getPeakBackBufferSizeBytes())/1024/1024, 0, 'f', 1));
}

#include "ConnectedServersDock.hxx"
#include "ui/ui_ConnectedServersDock.h"

ConnectedServersDock::ConnectedServersDock(QWidget *parent, const RenderServerConnections & serverConnections) :
    QDockWidget(parent),
    m_model(ConnectedServersTableModel(this, serverConnections)),
    ui(new Ui::ConnectedServersDock)
{
    ui->setupUi(this);
    this->setAllowedAreas(Qt::BottomDockWidgetArea);
    this->setMinimumSize(QSize(0,180));
    this->setFixedHeight(180);
    ui->tableView->setModel(&m_model);

    ui->tableView->horizontalHeader()->show();
    ui->tableView->horizontalHeader()->setResizeMode(QHeaderView::ResizeToContents);
    ui->tableView->verticalHeader()->setDefaultSectionSize(100);

    ui->tableView->verticalHeader()->hide();
    ui->tableView->verticalHeader()->setStyleSheet("background:black;color:white;");
    ui->tableView->verticalHeader()->setDefaultSectionSize(20); // height of a row
}

ConnectedServersDock::~ConnectedServersDock()
{
    delete ui;
}

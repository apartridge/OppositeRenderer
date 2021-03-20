#ifndef RENDERSERVERSDOCK_H
#define RENDERSERVERSDOCK_H

#include <QDockWidget>

namespace Ui {
class RenderServersSummaryDock;
}

class DistributedApplication;

class RenderServersSummaryDock : public QDockWidget
{
    Q_OBJECT
    
public:
    explicit RenderServersSummaryDock(QWidget *parent, DistributedApplication & application);
    ~RenderServersSummaryDock();
    
public slots:
    void onServersInfoUpdated();

signals:
    void actionConnectToNewRenderServer();


private:
    Ui::RenderServersSummaryDock *ui;
    DistributedApplication & m_application;
};

#endif // RENDERSERVERSDOCK_H

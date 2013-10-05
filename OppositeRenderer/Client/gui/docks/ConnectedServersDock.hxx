#pragma once
#include <QDockWidget>
#include "gui_models/ConnectedServersTableModel.hxx"

namespace Ui
{
class ConnectedServersDock;
}

class RenderServerConnections;

class ConnectedServersDock : public QDockWidget
{
    Q_OBJECT
    
public:
    explicit ConnectedServersDock(QWidget *parent, const RenderServerConnections & serverConnections );
    ~ConnectedServersDock();

private:
    Ui::ConnectedServersDock *ui;
    ConnectedServersTableModel m_model;
};
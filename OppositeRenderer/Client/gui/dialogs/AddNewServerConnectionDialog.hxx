#pragma once

#include <QDialog>
#include <QTcpSocket>
#include <QHostAddress>
#include "client/RenderServerConnection.hxx"

namespace Ui
{
    class AddNewServerConnectionDialog;
}

class QTimer;

class AddNewServerConnectionDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit AddNewServerConnectionDialog(QWidget *parent, QThread* tcpSocketThread);
    ~AddNewServerConnectionDialog();

signals:
    void hasNewServerConnectionSocket(QTcpSocket*);

private slots:
    void onFormSubmit();
    void onHostConnectionError();
    void onConnectedToHost();
    void onHostDataAvailable();
    void onWaitForGreetingError();

private:
    void showFormInitialState();
    void setError(const QString&);
    void socketDisconnectAndWait();
    Ui::AddNewServerConnectionDialog *ui;
    QTcpSocket* m_socket;
    QTimer* m_timerWaitForGreeting;
};
#include "AddNewServerConnectionDialog.hxx"
#include "ui/ui_AddNewServerConnectionDialog.h"
#include <QTimer>
#include <QMessageBox>

AddNewServerConnectionDialog::AddNewServerConnectionDialog(QWidget *parent, QThread* tcpSocketThread) :
    QDialog(parent),
    ui(new Ui::AddNewServerConnectionDialog)
{
    ui->setupUi(this);
    ui->errorLabel->setText("");
    connect(ui->connectButton, SIGNAL(clicked()), this, SLOT(onFormSubmit()));
    connect(this, SIGNAL(finished(int)), this, SLOT(deleteLater()));

    m_socket = new QTcpSocket(NULL);

    connect(m_socket, SIGNAL(connected()), this, SLOT(onConnectedToHost()));
    connect(m_socket, SIGNAL(error(QAbstractSocket::SocketError)), this, SLOT(onHostConnectionError()));
    connect(m_socket, SIGNAL(readyRead()), this, SLOT(onHostDataAvailable()));
    
    m_timerWaitForGreeting = new QTimer();
    connect(m_timerWaitForGreeting, SIGNAL(timeout()), this, SLOT(onWaitForGreetingError()));

    showFormInitialState();
}

AddNewServerConnectionDialog::~AddNewServerConnectionDialog()
{
    delete ui;
    delete m_timerWaitForGreeting;

    // m_socket is set to NULL if we have emitted hasNewServerConnection
    delete m_socket;
}

void AddNewServerConnectionDialog::showFormInitialState()
{
    m_timerWaitForGreeting->stop();
    ui->connectButton->setEnabled(true);
    ui->connectButton->setText("Connect to server");
    ui->formWidget->show();
}

void AddNewServerConnectionDialog::setError(const QString& error)
{
    ui->errorLabel->setText(error);
}

void AddNewServerConnectionDialog::onFormSubmit()
{
    setError("");
    ui->connectButton->setText("Connecting to server... Please wait.");
    ui->connectButton->setEnabled(false);

    QString serverHostAddressString = ui->ipValue->text();
    bool portNumberConversionOk;
    quint16 portNumber = ui->portValue->text().toUShort(&portNumberConversionOk);

    if(!portNumberConversionOk)
    {
        setError(QString("Invalid port number %1.").arg(portNumber));
        showFormInitialState();
    }
    else
    {
        QHostAddress serverAddress(serverHostAddressString);
        /*if(m_socket.state() == QAbstractSocket::ConnectingState || m_socket.state() == QAbstractSocket::HostLookupState 
                || m_socket.state() == QAbstractSocket::ConnectedState)
        {
            m_socket.disconnect();
            m_socket.waitForDisconnected();
        }*/

        m_socket->connectToHost(serverAddress, portNumber);
    }
}

void AddNewServerConnectionDialog::onHostConnectionError()
{
    setError(QString("Could not connect to host %1 on port %2. Please check if the server is available.")
                .arg(ui->ipValue->text()).arg(ui->portValue->text()));
    m_socket->disconnectFromHost();
    showFormInitialState();
}

void AddNewServerConnectionDialog::onConnectedToHost()
{
    m_timerWaitForGreeting->setInterval(7500);
    m_timerWaitForGreeting->setSingleShot(true);
    m_timerWaitForGreeting->start();
}

void AddNewServerConnectionDialog::onHostDataAvailable()
{
    m_timerWaitForGreeting->stop();

    // Check if the available data from the server (greeting) is correct
    // And get the compute device name from the client

    char readFromSocket[8];
    QTcpSocket* socket = m_socket;
    if(socket != NULL)
    {
        QDataStream stream(socket);
        QString greeting;
        stream >> greeting;

        if(greeting.startsWith("RSHELLO"))
        {
            //QString computeDeviceName = greeting.right(greeting.size()-8);
            disconnect(m_socket); // Prevent any new signals to this socket

            // Pass ownership of socket and RenderServerConnection to the recipient of this signal
            emit hasNewServerConnectionSocket(m_socket);

            m_socket = NULL; // Set m_socket to null to pass ownership of m_socket to receiver of signal
            this->accept();
        }
        else
        {
            setError("Connection established, but did not get expected greeting message! May be currently connected?");
            socketDisconnectAndWait();
            showFormInitialState();
        }
    }
}

void AddNewServerConnectionDialog::onWaitForGreetingError()
{
    setError("Connection established, but timed out waiting for greeting message!");
    socketDisconnectAndWait();
    showFormInitialState();
}

/*
Disconnect the socket and wait until it is disconnected.
*/

void AddNewServerConnectionDialog::socketDisconnectAndWait()
{
    m_socket->disconnectFromHost();
    if(m_socket->state() != QAbstractSocket::UnconnectedState)
    {
        m_socket->waitForDisconnected();
    }
}
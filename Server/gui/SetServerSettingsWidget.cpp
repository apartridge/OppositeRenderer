/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "SetServerSettingsWidget.hxx"
#include "ui/ui_SetServerSettingsWidget.h"
#include <QMessageBox>

SetServerSettingsWidget::SetServerSettingsWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SetServerSettingsWidget)
{
    ui->setupUi(this);
    connect(ui->startServerButton, SIGNAL(clicked()), this, SLOT(onStartServerButtonClicked()));
    connect(ui->portNumber, SIGNAL(returnPressed()), ui->startServerButton, SIGNAL(clicked()));

}

SetServerSettingsWidget::~SetServerSettingsWidget()
{
    delete ui;
}

void SetServerSettingsWidget::onStartServerButtonClicked()
{
    bool ok;
    m_portNumber = ui->portNumber->text().toUShort(&ok);
    if(m_portNumber > 0 && ok)
    {
        emit startServerFormSubmitted();
    }
    else
    {
        QMessageBox::information(this, "Port number is not valid", 
                        "Please select a port number between 1 and 65535. Please make sure it's not in use on your system.");
    }
}

ushort SetServerSettingsWidget::getPortNumber() const
{
    return m_portNumber;
}
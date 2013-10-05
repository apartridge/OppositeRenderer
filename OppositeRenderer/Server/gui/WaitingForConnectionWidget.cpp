/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "WaitingForConnectionWidget.hxx"
#include "ui/ui_WaitingForConnectionWidget.h"

WaitingForConnectionWidget::WaitingForConnectionWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::WaitingForConnectionWidget)
{
    ui->setupUi(this);
}

WaitingForConnectionWidget::~WaitingForConnectionWidget()
{
    delete ui;
}

void WaitingForConnectionWidget::setPortNumber( QString& string )
{
    ui->portNumberLabel->setText(string);
}
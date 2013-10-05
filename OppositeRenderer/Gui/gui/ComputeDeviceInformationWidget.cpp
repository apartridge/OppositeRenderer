/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "ComputeDeviceInformationWidget.hxx"
#include "ComputeDeviceInformationWidgetTabPage.hxx"
#include "ComputeDevice.h"
#include "ComputeDeviceRepository.h"

ComputeDeviceInformationWidget::ComputeDeviceInformationWidget(QWidget* parent, ComputeDeviceRepository & computeDeviceRepo) : 
    QWidget(parent)
{
    setupUi(this);
    setupGpuDeviceTabPages(computeDeviceRepo);
    //connect(this, SIGNAL(hasChangedDeviceConfiguration()), parent, SLOT(onHasChangedDeviceConfiguration()));
}

void ComputeDeviceInformationWidget::setupGpuDeviceTabPages(ComputeDeviceRepository & computeDeviceRepo)
{
    std::vector<ComputeDevice> & computeDevices = computeDeviceRepo.getComputeDevices();

    this->setWindowTitle(QString("Configure Compute Devices (") + QString::number(computeDevices.size()) + QString(" devices found)"));

    for(int i = 0; i < computeDevices.size(); i++)
    {
        ComputeDeviceInformationWidgetTabPage* deviceTabPage = new ComputeDeviceInformationWidgetTabPage(this, computeDevices.at(i));
        QString computeDevName = computeDevices.at(i).getName();
        connect(deviceTabPage, SIGNAL(hasSelectedComputeDevice(ComputeDevice*)), this, SLOT(onHasSelectedComputeDevice(ComputeDevice*)));
        this->tabWidget->addTab(deviceTabPage, computeDevName);
        this->m_tabPages.push_back(deviceTabPage);
    }
}

void ComputeDeviceInformationWidget::onHasSelectedComputeDevice( ComputeDevice* device)
{
    emit hasSelectedComputeDevice(device);
}
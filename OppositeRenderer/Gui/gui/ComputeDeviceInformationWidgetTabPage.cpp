/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "ComputeDeviceInformationWidgetTabPage.hxx"
#include <QDebug>
#include "ComputeDevice.h"

ComputeDeviceInformationWidgetTabPage::ComputeDeviceInformationWidgetTabPage(QWidget* parent, ComputeDevice & device)
    : m_device(device),
      QWidget(parent)
{
    setupUi(this);
    
    GlobalMemory->setText(QString::number(device.getGlobalMemoryKB()/1024) + QString(" MB"));
    ConstantMemory->setText(QString::number(device.getConstantMemoryKB()) + QString(" KB"));
    L2CacheMemory->setText(QString::number(device.getL2CacheMemoryKB()) + QString(" KB"));
    SharedMemorySM->setText(QString::number(device.getSharedMemoryPerBlockKB()) + QString(" KB"));
    NumSM->setText(QString::number(device.getMultiProcessorCount()));
    ThreadsPerSM->setText(QString::number(device.getMaxThreadsPerMultiProcessor()));
    RegisterMemorySM->setText(QString::number(device.getRegisterMemoryPerBlockKB()) + QString(" KB"));
    RegistersSM->setText(QString::number(device.getRegistersPerBlock()));
    MemClockFq->setText(QString::number(device.getMemoryClockFrequencyKHz()/1024)+QString(" MhZ"));
    PrcClockFrq->setText(QString::number(device.getClockFrequencyKHz()/1024)+QString(" MhZ"));
    WarpSize->setText(QString::number(device.getWarpSize()));
    ComputeCapability->setText(QString(device.getComputeCapability()));
    MaxBlockDim->setText(QString::number(device.getMaxBlockDimensionX()) + QString(" x ") + QString::number(device.getMaxBlockDimensionY())
                        + QString(" x ") + QString::number(device.getMaxBlockDimensionZ()));
    MaxGridDim->setText(QString::number(device.getMaxGridDimensionX()) + QString(" x ") + QString::number(device.getMaxGridDimensionY())
        + QString(" x ") + QString::number(device.getMaxGridDimensionZ()));
    MaxThreadsPerBlock->setText(QString::number(device.getMaxThreadsPerBlock()));
    DeviceId->setText(QString::number(device.getDeviceId()));
    PCIBusId->setText(QString::number(device.getPCIBusId()));
    MemoryBusWidth->setText(QString::number(device.getMemoryBusWidth())+QString(" bits"));
    UnifiedAddressing->setText(QString(device.getUnifiedAddressing() ? "Yes" : "No"));
    connect(this->buttonSelectDevice, SIGNAL(clicked()), this, SLOT(onButtonSelectComputeDevice()));
}

void ComputeDeviceInformationWidgetTabPage::onButtonSelectComputeDevice()
{
    emit hasSelectedComputeDevice(&m_device);
}

ComputeDevice& ComputeDeviceInformationWidgetTabPage::getComputeDevice()
{
    return this->m_device;
}

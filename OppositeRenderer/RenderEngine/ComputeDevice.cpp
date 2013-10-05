/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "ComputeDevice.h"
#include "cuda_runtime_api.h"
#include <cstring>
#include <cstdio>

ComputeDevice::ComputeDevice()
{
    m_enabled = true;
}

ComputeDevice ComputeDevice::fromCudaDeviceProperties(const cudaDeviceProp& devProp, int cudaDeviceId )
{
    ComputeDevice q;
    q.m_globalMemoryKB = (unsigned int)devProp.totalGlobalMem/1024;
    q.m_constantMemoryKB = (unsigned int)devProp.totalConstMem/1024;
    q.m_warpSize = devProp.warpSize;
    q.m_memoryClockFrequencyKHz = (unsigned int)devProp.memoryClockRate;
    q.m_clockFrequencyKHz = (unsigned int)devProp.clockRate;
    q.m_multiProcessorCount = (unsigned int)devProp.multiProcessorCount;
    q.m_maxThreadsPerMultiProcessor = (unsigned int)devProp.maxThreadsPerMultiProcessor;
    q.m_maxThreadsPerBlock = (unsigned int)devProp.maxThreadsPerBlock;
    q.m_L2CacheMemoryKB = (unsigned int)devProp.l2CacheSize/1024;
    q.m_sharedMemoryPerBlockKB = (unsigned int)devProp.sharedMemPerBlock/1024;
    q.m_registersPerBlock = (unsigned int)devProp.regsPerBlock;
    q.m_registerMemoryPerBlockKB = (unsigned int)devProp.regsPerBlock*4/1024;
    strncpy_s(q.m_name, devProp.name, 95);
    sprintf(q.m_computeCapability, "%d.%d", devProp.major, devProp.minor);
    q.m_deviceId = cudaDeviceId;
    q.m_maxGridDimensionX = devProp.maxGridSize[0];
    q.m_maxGridDimensionY = devProp.maxGridSize[1];
    q.m_maxGridDimensionZ = devProp.maxGridSize[2];
    q.m_PCIBusId = devProp.pciBusID;
    q.m_PCIDeviceId = devProp.pciDeviceID;
    q.m_PCIDomainId = devProp.pciDomainID;
    q.m_memoryBusWidth = devProp.memoryBusWidth;
    q.m_TCCDriver = devProp.tccDriver==1;
    q.m_unifiedAddressing = devProp.unifiedAddressing==1;
    q.m_maxBlockDimensionX = devProp.maxThreadsDim[0];
    q.m_maxBlockDimensionY = devProp.maxThreadsDim[1];
    q.m_maxBlockDimensionZ = devProp.maxThreadsDim[2];

    return q;
}

unsigned int ComputeDevice::getGlobalMemoryKB()  const
{
    return m_globalMemoryKB;
}

unsigned int ComputeDevice::getConstantMemoryKB()  const
{
    return m_constantMemoryKB;
}

unsigned int ComputeDevice::getWarpSize()  const
{
    return m_warpSize;
}

const char* ComputeDevice::getName() const
{
    return (const char*) this->m_name;
}

bool ComputeDevice::isEnabled() const
{
    return m_enabled;
}

unsigned int ComputeDevice::getMemoryClockFrequencyKHz() const
{
    return m_memoryClockFrequencyKHz;
}

unsigned int ComputeDevice::getClockFrequencyKHz() const
{
    return m_clockFrequencyKHz;
}

unsigned int ComputeDevice::getMultiProcessorCount() const
{
    return m_multiProcessorCount;
}

unsigned int ComputeDevice::getMaxThreadsPerMultiProcessor() const
{
    return m_maxThreadsPerMultiProcessor;
}

unsigned int ComputeDevice::getL2CacheMemoryKB() const
{
    return m_L2CacheMemoryKB;
}

unsigned int ComputeDevice::getSharedMemoryPerBlockKB() const
{
    return m_sharedMemoryPerBlockKB;
}

unsigned int ComputeDevice::getRegistersPerBlock() const
{
    return m_registersPerBlock;
}

const char* ComputeDevice::getComputeCapability() const
{
    return m_computeCapability;
}

unsigned int ComputeDevice::getMaxThreadsPerBlock() const
{
    return m_maxThreadsPerBlock;
}

unsigned int ComputeDevice::getRegisterMemoryPerBlockKB() const
{
    return m_registerMemoryPerBlockKB;
}

unsigned int ComputeDevice::getMaxGridDimensionX() const
{
    return m_maxGridDimensionX;
}

unsigned int ComputeDevice::getMaxGridDimensionY() const
{
    return m_maxGridDimensionY;
}

unsigned int ComputeDevice::getMaxGridDimensionZ() const
{
    return m_maxGridDimensionZ;
}

unsigned int ComputeDevice::getMaxBlockDimensionX() const
{
    return m_maxBlockDimensionX;
}

unsigned int ComputeDevice::getMaxBlockDimensionY() const
{
    return m_maxBlockDimensionY;
}

unsigned int ComputeDevice::getMaxBlockDimensionZ() const
{
    return m_maxBlockDimensionZ;
}

void ComputeDevice::setEnabled( bool enabled )
{
    m_enabled = enabled;
}

int ComputeDevice::getDeviceId() const
{
    return m_deviceId;
}

int ComputeDevice::getPCIDeviceId() const
{
    return m_PCIDeviceId;
}

int ComputeDevice::getPCIBusId() const
{
    return m_PCIBusId;
}

int ComputeDevice::getPCIDomainId() const
{
    return m_PCIDomainId;
}

int ComputeDevice::getMemoryBusWidth() const
{
    return m_memoryBusWidth;
}

bool ComputeDevice::getTCCDriver() const
{
    return m_TCCDriver;
}

bool ComputeDevice::getUnifiedAddressing() const
{
    return m_unifiedAddressing;
}

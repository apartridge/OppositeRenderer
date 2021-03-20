/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include "render_engine_export_api.h"

struct cudaDeviceProp;

class ComputeDevice
{
public:
    ComputeDevice();
    static ComputeDevice fromCudaDeviceProperties(const cudaDeviceProp& devProp, int );
    RENDER_ENGINE_EXPORT_API unsigned int getGlobalMemoryKB() const;
    RENDER_ENGINE_EXPORT_API unsigned int getConstantMemoryKB() const;
    RENDER_ENGINE_EXPORT_API unsigned int getMemoryClockFrequencyKHz() const;
    RENDER_ENGINE_EXPORT_API unsigned int getClockFrequencyKHz() const;
    RENDER_ENGINE_EXPORT_API int getDeviceId() const;
    RENDER_ENGINE_EXPORT_API int getPCIDeviceId() const;
    RENDER_ENGINE_EXPORT_API int getPCIBusId() const;
    RENDER_ENGINE_EXPORT_API int getPCIDomainId() const;
    RENDER_ENGINE_EXPORT_API int getMemoryBusWidth() const;
    RENDER_ENGINE_EXPORT_API bool getTCCDriver() const;
    RENDER_ENGINE_EXPORT_API bool getUnifiedAddressing() const;
    RENDER_ENGINE_EXPORT_API const char* getName() const;
    RENDER_ENGINE_EXPORT_API bool isEnabled() const;
    RENDER_ENGINE_EXPORT_API void setEnabled(bool enabled);
    RENDER_ENGINE_EXPORT_API unsigned int getWarpSize() const;
    RENDER_ENGINE_EXPORT_API unsigned int getMultiProcessorCount() const;
    RENDER_ENGINE_EXPORT_API unsigned int getMaxThreadsPerMultiProcessor() const;
    RENDER_ENGINE_EXPORT_API unsigned int getL2CacheMemoryKB() const;
    RENDER_ENGINE_EXPORT_API unsigned int getSharedMemoryPerBlockKB() const;
    RENDER_ENGINE_EXPORT_API unsigned int getRegistersPerBlock() const;
    RENDER_ENGINE_EXPORT_API unsigned int getMaxThreadsPerBlock() const;
    RENDER_ENGINE_EXPORT_API unsigned int getRegisterMemoryPerBlockKB() const;
    RENDER_ENGINE_EXPORT_API unsigned int getMaxGridDimensionX() const;
    RENDER_ENGINE_EXPORT_API unsigned int getMaxGridDimensionY() const;
    RENDER_ENGINE_EXPORT_API unsigned int getMaxGridDimensionZ() const;
    RENDER_ENGINE_EXPORT_API unsigned int getMaxBlockDimensionX() const;
    RENDER_ENGINE_EXPORT_API unsigned int getMaxBlockDimensionY() const;
    RENDER_ENGINE_EXPORT_API unsigned int getMaxBlockDimensionZ() const;
    RENDER_ENGINE_EXPORT_API const char* getComputeCapability() const;

private:
    bool m_enabled;
    unsigned int m_globalMemoryKB;
    unsigned int m_constantMemoryKB;
    unsigned int m_clockFrequencyKHz;
    unsigned int m_memoryClockFrequencyKHz;
    unsigned int m_warpSize;
    unsigned int m_multiProcessorCount;
    unsigned int m_maxThreadsPerMultiProcessor;
    unsigned int m_maxThreadsPerBlock;
    unsigned int m_L2CacheMemoryKB;
    unsigned int m_sharedMemoryPerBlockKB;
    unsigned int m_registersPerBlock;
    unsigned int m_registerMemoryPerBlockKB;
    unsigned int m_maxGridDimensionX;
    unsigned int m_maxGridDimensionY;
    unsigned int m_maxGridDimensionZ;
    unsigned int m_maxBlockDimensionX;
    unsigned int m_maxBlockDimensionY;
    unsigned int m_maxBlockDimensionZ;
    int m_deviceId;
    int m_PCIDeviceId;
    int m_PCIBusId;
    int m_PCIDomainId;
    int m_memoryBusWidth;
    bool m_TCCDriver;
    bool m_unifiedAddressing;
    char m_name[100];
    char m_computeCapability[5];
};
/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "ComputeDeviceRepository.h"
#include <cuda.h>
#include <cuda_runtime.h>

ComputeDeviceRepository::ComputeDeviceRepository(void)
{
    int numDevs;
    cudaGetDeviceCount(&numDevs);

    for(int i = 0; i < numDevs; i++)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        ComputeDevice device = ComputeDevice::fromCudaDeviceProperties(devProp, i);
        this->m_computeDevices.push_back(device);
    }
}

ComputeDeviceRepository::~ComputeDeviceRepository(void)
{

}

std::vector<ComputeDevice> & ComputeDeviceRepository::getComputeDevices()
{
    return m_computeDevices;
}

/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma  once

template<typename T>
static T* getDevicePtr(optix::Buffer & buffer, int deviceNumber)
{
    CUdeviceptr d;
    buffer->getDevicePointer(deviceNumber, &d);
    return (T*)d;
}

template<typename T>
static thrust::device_ptr<T> getThrustDevicePtr(optix::Buffer & buffer, int deviceNumber)
{
    return thrust::device_pointer_cast(getDevicePtr<T>(buffer, deviceNumber));
}

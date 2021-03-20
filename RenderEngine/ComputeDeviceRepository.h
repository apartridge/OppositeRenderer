/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
#include <vector>
#include "ComputeDevice.h"
#include "render_engine_export_api.h"

class ComputeDeviceRepository
{
public:
    RENDER_ENGINE_EXPORT_API ComputeDeviceRepository(void);
    RENDER_ENGINE_EXPORT_API ~ComputeDeviceRepository(void);
    RENDER_ENGINE_EXPORT_API static ComputeDeviceRepository& get();
    RENDER_ENGINE_EXPORT_API std::vector<ComputeDevice> & getComputeDevices();

private: 
    std::vector<ComputeDevice> m_computeDevices;
};

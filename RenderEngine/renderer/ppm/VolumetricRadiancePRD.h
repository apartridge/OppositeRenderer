/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once
struct VolumetricRadiancePRD
{
    float sigma_t;
    float sigma_s;
    optix::float3 radiance;
    unsigned int numHits;
};

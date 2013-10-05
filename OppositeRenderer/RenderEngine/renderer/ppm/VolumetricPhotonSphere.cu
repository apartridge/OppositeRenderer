/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include <optix_world.h>
#include "Photon.h"

#if ENABLE_PARTICIPATING_MEDIA

using namespace optix;

rtDeclareVariable(float3, photonPosition, attribute photonPosition,); 
rtDeclareVariable(float3, photonPower, attribute photonPower,); 
rtDeclareVariable(uint, photonId, attribute photonId,); 
rtDeclareVariable(float, volumetricRadius, ,); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay,);
rtBuffer<Photon> photonsBuffer; 

// Intersection code in part from NVIDIA's Optix SDK

RT_PROGRAM void intersect(int primIdx)
{
    Photon & photon = photonsBuffer[primIdx];

    float3 O = ray.origin - photon.position;
    float b = dot(O, ray.direction );
    float c = dot(O, O) - volumetricRadius*volumetricRadius;
    float disc = b*b - c;
    if( disc > 0.0f )
    {
        float sdisc = sqrtf( disc );
        float root1 = (-b - sdisc);
        bool check_second = true;

        if( rtPotentialIntersection(root1))
        {
            photonId = primIdx;
            photonPosition = photon.position;
            photonPower = photon.power*photon.numDeposits;
            if( rtReportIntersection( 0 ) )
                check_second = false;
        } 

        if( check_second )
        {
            float root2 = (-b + sdisc);
            if( rtPotentialIntersection(root2))
            {
                photonId = primIdx;
                photonPosition = photon.position;
                photonPower = photon.power*photon.numDeposits;
                rtReportIntersection( 0 );
            }
        }
    }
}

RT_PROGRAM void boundingBox(int primIdx, float result[6])
{
    Photon & photon = photonsBuffer[primIdx];
    if(fmaxf(photon.power) > 0)
    {
        const float3 radius3 = make_float3(volumetricRadius);
        float3 min_ = photon.position - radius3;
        float3 max_ = photon.position + radius3;
        result[0] = min_.x;
        result[1] = min_.y;
        result[2] = min_.z;
        result[3] = max_.x;
        result[4] = max_.y;
        result[5] = max_.z;
    }
    else
    {
        result[0] = 0.0f;
        result[1] = 0.0f;
        result[2] = 0.0f;
        result[3] = 0.0f;
        result[4] = 0.0f;
        result[5] = 0.0f;
    }
}

#endif
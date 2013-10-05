/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float3, center, , );
rtDeclareVariable(float, radius, , );
rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, ); 
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void intersect(int primIdx)
{
    float3 O = ray.origin - center;
    float b = dot( O, ray.direction );
    float c = dot( O, O ) - radius*radius;
    float disc = b*b - c;
    if( disc > 0.0f )
    {
        float sdisc = sqrtf( disc );
        float root1 = (-b - sdisc);
        bool check_second = true;
        if( rtPotentialIntersection( root1 ) ) {
            shadingNormal = geometricNormal = (O + root1*ray.direction) / radius;
            if( rtReportIntersection( 0 ) )
                check_second = false;
        } 
        if( check_second ) {
            float root2 = (-b + sdisc);
            if( rtPotentialIntersection( root2 ) ) {
                shadingNormal = geometricNormal = (O + root2*ray.direction) / radius;
                rtReportIntersection( 0 );
            }
        }
    }
}

RT_PROGRAM void boundingBox(int, float result[6])
{
    const float3 radius3 = make_float3(radius);

    float3 min_ = center - radius3;
    float3 max_ = center + radius3;
    result[0] = min_.x;
    result[1] = min_.y;
    result[2] = min_.z;
    result[3] = max_.x;
    result[4] = max_.y;
    result[5] = max_.z;
}


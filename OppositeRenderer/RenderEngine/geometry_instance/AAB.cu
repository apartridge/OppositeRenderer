/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include <optix_world.h>
#include "renderer/RayType.h"

using namespace optix;

rtDeclareVariable(float3, cuboidMin, , );
rtDeclareVariable(float3, cuboidMax, , );
rtDeclareVariable(float3, geometricNormal, attribute geometricNormal, ); 
rtDeclareVariable(float3, shadingNormal, attribute shadingNormal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

namespace materialProperties
{
    rtDeclareVariable(int, participatingMedium, , );
}    

RT_PROGRAM void intersect(int primIdx)
{
    // X
    int intersectionAxisNear = 0; //X=0, Y=1, Z=2
    int intersectionAxisFar = 0; //X=0, Y=1, Z=2

    float divx = 1 / ray.direction.x;
    float tNear, tFar;
    if (divx >= 0)
    {
        tNear = (cuboidMin.x- ray.origin.x) * divx;
        tFar = (cuboidMax.x - ray.origin.x) * divx;
    }
    else
    {
        tNear = (cuboidMax.x - ray.origin.x) * divx;
        tFar = (cuboidMin.x - ray.origin.x) * divx;
    }

    if(tFar < tNear)
    {
        return;
    }

    // Y

    float divy = 1 / ray.direction.y;
    float tyNear, tyFar;
    if (divy >= 0)
    {
        tyNear = (cuboidMin.y- ray.origin.y) * divy;
        tyFar = (cuboidMax.y - ray.origin.y) * divy;
    }
    else {
        tyNear = (cuboidMax.y - ray.origin.y) * divy;
        tyFar = (cuboidMin.y - ray.origin.y) * divy;
    }

    if(tyNear > tNear)
    {
        tNear = tyNear;
        intersectionAxisNear = 1;
    }

    if(tyFar < tFar)
    {
        tFar = tyFar;
        intersectionAxisFar = 1;
    }

    if(tFar < tNear)
    {
        return;
    }

    // Z

    float divz = 1 / ray.direction.z;
    float tzNear, tzFar;
    if (divz >= 0)
    {
        tzNear = (cuboidMin.z- ray.origin.z) * divz;
        tzFar = (cuboidMax.z - ray.origin.z) * divz;
    }
    else {
        tzNear = (cuboidMax.z - ray.origin.z) * divz;
        tzFar = (cuboidMin.z - ray.origin.z) * divz;
    }


    if(tzNear > tNear)
    {
        tNear = tzNear;
        intersectionAxisNear = 2;
    }

    if(tzFar < tFar)
    {
        tFar = tzFar;
        intersectionAxisFar = 2;
    }

    if(tFar < tNear)
    {
        return;
    }

    float t = tNear;
    int intersectionAxis = intersectionAxisNear;
    
    // If we are inside the cuboid, then we intersect at tFar. Furthermore, the normal
    // of faces will point the same way as the ray direction's component on the intersection axis.
    // If we are outside when we hit, then the normal points in the opposite direction as the ray direction's component
    // in the intersection axis.

    float normalVsRayDirection = -1;
    // Take special action if we are inside the volume
    if(tNear < 0.01f && tFar > 0.0f)
    {
        if((ray.ray_type == RayType::PHOTON || ray.ray_type == RayType::RADIANCE || ray.ray_type == RayType::PARTICIPATING_MEDIUM_TRANSMISSION) )
        {
            //
            intersectionAxis = intersectionAxisFar;
            t = 0.000115; // must be larger than 0.0001 for now
        }
        else
        {
            intersectionAxis = intersectionAxisFar;
            t = tFar;
            normalVsRayDirection = 1;
        }
    }

    if(rtPotentialIntersection(t))
    {
        float3 hitpoint = ray.origin + t*ray.direction;

        if(intersectionAxis == 0)
        {
            geometricNormal = make_float3( ray.direction.x >= 0 ? normalVsRayDirection : -normalVsRayDirection , 0, 0);
        }
        else if(intersectionAxis == 1)
        {
            geometricNormal = make_float3( 0, ray.direction.y >= 0 ? normalVsRayDirection : -normalVsRayDirection , 0);
        }
        else
        {
            geometricNormal = make_float3( 0, 0, ray.direction.z >= 0 ? normalVsRayDirection : -normalVsRayDirection);
        }

        shadingNormal = geometricNormal;
        rtReportIntersection(0);
    }
}

RT_PROGRAM void boundingBox(int, float result[6])
{
    result[0] = cuboidMin.x;
    result[1] = cuboidMin.y;
    result[2] = cuboidMin.z;
    result[3] = cuboidMax.x;
    result[4] = cuboidMax.y;
    result[5] = cuboidMax.z;
}

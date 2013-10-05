/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#include "OptixRenderer.h"
#include "renderer/ppm/Photon.h"
#include "config.h"
#include "select.h"

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_KD_TREE_CPU

inline RT_HOSTDEVICE int max_component(optix::float3 a)
{
    if(a.x > a.y && a.x  > a.z)
    {
        return 0;
    }
    else if(a.y > a.z)
    {
        return 1;
    }
    return 2;
}

static void buildKDTree( Photon* photons, int start, int end, int depth, Photon* kd_tree, int current_root,
    optix::float3 bbmin, optix::float3 bbmax)
{
    // If we have zero photons, this is a NULL node
    if( end - start == 0 ) {
        kd_tree[current_root].axis = PPM_NULL;
        kd_tree[current_root].power = optix::make_float3( 0.0f );
        return;
    }

    // If we have a single photon
    if( end - start == 1 ) {
        photons[start].axis = PPM_LEAF;
        kd_tree[current_root] = (photons[start]);
        return;
    }

    // Choose axis to split on
    int axis;

    optix::float3 diag = bbmax-bbmin;
    axis = max_component(diag);

    int median = (start+end) / 2;
    Photon* start_addr = &(photons[start]);
    switch( axis ) {
    case 0:
        select<Photon, 0>( start_addr, 0, end-start-1, median-start );
        photons[median].axis = PPM_X;
        break;
    case 1:
        select<Photon, 1>( start_addr, 0, end-start-1, median-start );
        photons[median].axis = PPM_Y;
        break;
    case 2:
        select<Photon, 2>( start_addr, 0, end-start-1, median-start );
        photons[median].axis = PPM_Z;
        break;
    }
    optix::float3 rightMin = bbmin;
    optix::float3 leftMax  = bbmax;
    optix::float3 midPoint = (photons[median]).position;
    switch( axis ) {
    case 0:
        rightMin.x = midPoint.x;
        leftMax.x  = midPoint.x;
        break;
    case 1:
        rightMin.y = midPoint.y;
        leftMax.y  = midPoint.y;
        break;
    case 2:
        rightMin.z = midPoint.z;
        leftMax.z  = midPoint.z;
        break;
    }

    kd_tree[current_root] = (photons[median]);
    buildKDTree( photons, start, median, depth+1, kd_tree, 2*current_root+1, bbmin,  leftMax );
    buildKDTree( photons, median+1, end, depth+1, kd_tree, 2*current_root+2, rightMin, bbmax );
}

void OptixRenderer::createPhotonKdTreeOnCPU()
{
    Photon* photons_host = reinterpret_cast<Photon*>( m_photons->map() );
    Photon* photonKdTree_host = reinterpret_cast<Photon*>( m_photonKdTree->map() );

    int numValidPhotons = NUM_PHOTONS >= m_photonKdTreeSize ? m_photonKdTreeSize : NUM_PHOTONS;

    for( unsigned int i = 0; i < numValidPhotons; ++i )
    {
        Photon a = photons_host[i];

        if(! (fmaxf(photons_host[i].power) > 0.0f))
        {
            photons_host[i] = photons_host[numValidPhotons-1];
            numValidPhotons--;
            i--;
        }
    }

    optix::float3 bbmin = optix::make_float3(0.0f);
    optix::float3 bbmax = optix::make_float3(0.0f);

    bbmin = optix::make_float3(  std::numeric_limits<float>::max() );
    bbmax = optix::make_float3( -std::numeric_limits<float>::max() );

    // Compute the bounds of the photons
    for(int i = 0; i < numValidPhotons; ++i)
    {
        optix::float3 position = (photons_host[i]).position;
        bbmin = fminf(bbmin, position);
        bbmax = fmaxf(bbmax, position);
    }

    // Now build KD tree
    buildKDTree( photons_host, 0, numValidPhotons, 0, photonKdTree_host, 0, bbmin, bbmax );

    m_numberOfPhotonsLastFrame = numValidPhotons;
    m_photonKdTree->unmap();
    m_photons->unmap();
}

#endif
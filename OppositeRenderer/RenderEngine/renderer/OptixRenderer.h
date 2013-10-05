/* 
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
*/

#pragma once

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "render_engine_export_api.h"
#include "math/AAB.h"

class ComputeDevice;
class RenderServerRenderRequestDetails;
class IScene;

class OptixRenderer
{
public:

    RENDER_ENGINE_EXPORT_API OptixRenderer();
    RENDER_ENGINE_EXPORT_API ~OptixRenderer();

    RENDER_ENGINE_EXPORT_API void initScene(IScene & scene);
    RENDER_ENGINE_EXPORT_API void initialize(const ComputeDevice & device);

    void createGpuDebugBuffers();

    RENDER_ENGINE_EXPORT_API void renderNextIteration(unsigned long long iterationNumber, unsigned long long localIterationNumber, 
        float PPMRadius, bool createOutput, const RenderServerRenderRequestDetails & details);
    RENDER_ENGINE_EXPORT_API void getOutputBuffer(void* data);
    RENDER_ENGINE_EXPORT_API unsigned int getWidth() const;
    RENDER_ENGINE_EXPORT_API unsigned int getHeight() const;
    RENDER_ENGINE_EXPORT_API unsigned int getScreenBufferSizeBytes() const;

    const static unsigned int NUM_PHOTONS;
    const static float PPM_INITIAL_RADIUS;
    const static unsigned int PHOTON_GRID_MAX_SIZE;
    RENDER_ENGINE_EXPORT_API const static unsigned int EMITTED_PHOTONS_PER_ITERATION;

private:
    void initDevice(const ComputeDevice & device);
    void compile();
    void loadObjGeometry( const std::string& filename, optix::Aabb& bbox );
    void initializeRandomStates();
    void createUniformGridPhotonMap(float ppmRadius);
    void initializeStochasticHashPhotonMap(float ppmRadius);
    void createPhotonKdTreeOnCPU();


    optix::Buffer m_outputBuffer;
    optix::Buffer m_photons;
    optix::Buffer m_photonKdTree;
    optix::Buffer m_hashmapOffsetTable;
    optix::Buffer m_photonsHashCells;
    optix::Buffer m_raytracePassOutputBuffer;
    optix::Buffer m_directRadianceBuffer;
    optix::Buffer m_indirectRadianceBuffer;
    optix::Group m_sceneRootGroup;
    optix::Buffer m_volumetricPhotonsBuffer;
    optix::Buffer m_lightBuffer;
    optix::Buffer m_randomStatesBuffer;

    unsigned int m_photonKdTreeSize;
    unsigned long long m_numberOfPhotonsLastFrame;
    float m_spatialHashMapCellSize;
    AAB m_sceneAABB;
    optix::uint3 m_gridSize;
    unsigned int m_spatialHashMapNumCells;

    unsigned int m_width;
    unsigned int m_height;

    bool m_initialized;

    const static unsigned int MAX_BOUNCES;
    const static unsigned int MAX_PHOTON_COUNT;
    const static unsigned int PHOTON_LAUNCH_WIDTH;
    const static unsigned int PHOTON_LAUNCH_HEIGHT;
   
    void resizeBuffers(unsigned int width, unsigned int height);
    void debugOutputPhotonTracing();
    optix::Context m_context;
    int m_optixDeviceOrdinal;

    // Volumetric
    optix::GeometryGroup m_volumetricPhotonsRoot;
};
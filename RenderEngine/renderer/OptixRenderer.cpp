/*
 * Copyright (c) 2013 Opposite Renderer
 * For the full copyright and license information, please view the LICENSE.txt
 * file that was distributed with this source code.
 */

#include "OptixRenderer.h"
#include "../ComputeDevice.h"
#include "../clientserver/RenderServerRenderRequest.h"
#include "../config.h"
#include "../renderer/Hitpoint.h"
#include "../renderer/OptixEntryPoint.h"
#include "../renderer/RayType.h"
#include "../renderer/ppm/Photon.h"
#include "../scene/IScene.h"
#include "../util/ptxhelper.h"
#include "../util/time.h"
#include "Camera.h"
#include "RandomState.h"
//#include "../renderer/helpers/nsight.h"
#include <cuda_runtime_api.h>

#include <QThread>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_UNIFORM_GRID
const unsigned int OptixRenderer::PHOTON_GRID_MAX_SIZE = 100 * 100 * 100;
#else
const unsigned int OptixRenderer::PHOTON_GRID_MAX_SIZE = 0;
#endif

const unsigned int OptixRenderer::MAX_PHOTON_COUNT = MAX_PHOTONS_DEPOSITS_PER_EMITTED;
const unsigned int OptixRenderer::PHOTON_LAUNCH_WIDTH = 1024;
const unsigned int OptixRenderer::PHOTON_LAUNCH_HEIGHT = 1024;
// Ensure that NUM PHOTONS are a power of 2 for stochastic hash

const unsigned int OptixRenderer::EMITTED_PHOTONS_PER_ITERATION
    = OptixRenderer::PHOTON_LAUNCH_WIDTH * OptixRenderer::PHOTON_LAUNCH_HEIGHT;
const unsigned int OptixRenderer::NUM_PHOTONS
    = OptixRenderer::EMITTED_PHOTONS_PER_ITERATION * OptixRenderer::MAX_PHOTON_COUNT;

using namespace optix;

inline unsigned int pow2roundup(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

inline float max(float a, float b)
{
    return a > b ? a : b;
}

OptixRenderer::OptixRenderer()
    : m_initialized(false)
    , m_width(10)
    , m_height(10)
{
    try
    {
        m_context = optix::Context::create();
        if (!m_context)
        {
            throw std::runtime_error("Unable to create OptiX context.");
        }
    }
    catch (const optix::Exception& e)
    {
        throw std::runtime_error(e.getErrorString());
    }
    catch (const std::exception& e)
    {
        QString error = QString("Error during initialization of Optix: %1").arg(e.what());
        throw std::runtime_error(error.toStdString());
    }
}

OptixRenderer::~OptixRenderer()
{
    m_context->destroy();
    cudaDeviceReset();
}

void OptixRenderer::initialize(const ComputeDevice& device)
{
    if (m_initialized)
    {
        throw std::runtime_error("ERROR: Multiple OptixRenderer::initialize!\n");
    }

    initDevice(device);

    m_context->setRayTypeCount(static_cast<int>(RayType::NUM_RAY_TYPES));
    m_context->setEntryPointCount(OptixEntryPoint::NUM_PASSES);
    m_context->setStackSize(ENABLE_PARTICIPATING_MEDIA ? 3000 : 1596);

    m_context["maxPhotonDepositsPerEmitted"]->setUint(MAX_PHOTON_COUNT);
    m_context["ppmAlpha"]->setFloat(0);
    m_context["totalEmitted"]->setFloat(0.0f);
    m_context["iterationNumber"]->setFloat(0.0f);
    m_context["localIterationNumber"]->setUint(0);
    m_context["ppmRadius"]->setFloat(0.f);
    m_context["ppmRadiusSquared"]->setFloat(0.f);
    m_context["ppmRadiusSquaredNew"]->setFloat(0.f);
    m_context["ppmDefaultRadius2"]->setFloat(0.f);
    m_context["emittedPhotonsPerIteration"]->setUint(EMITTED_PHOTONS_PER_ITERATION);
    m_context["emittedPhotonsPerIterationFloat"]->setFloat(float(EMITTED_PHOTONS_PER_ITERATION));
    m_context["photonLaunchWidth"]->setUint(PHOTON_LAUNCH_WIDTH);
    m_context["participatingMedium"]->setUint(0);

    // An empty scene root node
    optix::Group group = m_context->createGroup();
    m_context["sceneRootObject"]->set(group);
    // optix::Acceleration acceleration = m_context->createAcceleration("Bvh", "Bvh");
    // group->setAcceleration(acceleration);

    // Display buffer

    // Ray Trace OptixEntryPoint Output Buffer

    m_raytracePassOutputBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
    m_raytracePassOutputBuffer->setFormat(RT_FORMAT_USER);
    m_raytracePassOutputBuffer->setElementSize(sizeof(Hitpoint));
    m_raytracePassOutputBuffer->setSize(m_width, m_height);
    m_context["raytracePassOutputBuffer"]->set(m_raytracePassOutputBuffer);

    // Ray OptixEntryPoint Generation Program

    {
        Program generatorProgram
            = m_context->createProgramFromPTXFile(getPtxFile("renderer/ppm/RayGeneratorPPM.ptx"), "generateRay");
        Program exceptionProgram
            = m_context->createProgramFromPTXFile(getPtxFile("renderer/ppm/RayGeneratorPPM.ptx"), "exception");
        Program missProgram
            = m_context->createProgramFromPTXFile(getPtxFile("renderer/ppm/RayGeneratorPPM.ptx"), "miss");

        m_context->setRayGenerationProgram(OptixEntryPoint::PPM_RAYTRACE_PASS, generatorProgram);
        m_context->setExceptionProgram(OptixEntryPoint::PPM_RAYTRACE_PASS, exceptionProgram);
        m_context->setMissProgram(static_cast<int>(RayType::RADIANCE), missProgram);
        m_context->setMissProgram(static_cast<int>(RayType::RADIANCE_IN_PARTICIPATING_MEDIUM), missProgram);
    }

    // Path Tracing ray generation
    {
        Program generatorProgram
            = m_context->createProgramFromPTXFile(getPtxFile("renderer/pt/RayGeneratorPT.ptx"), "generateRay");
        Program exceptionProgram
            = m_context->createProgramFromPTXFile(getPtxFile("renderer/pt/RayGeneratorPT.ptx"), "exception");
        Program missProgram = m_context->createProgramFromPTXFile(getPtxFile("renderer/pt/RayGeneratorPT.ptx"), "miss");
        m_context->setRayGenerationProgram(OptixEntryPoint::PT_RAYTRACE_PASS, generatorProgram);
        m_context->setExceptionProgram(OptixEntryPoint::PT_RAYTRACE_PASS, exceptionProgram);
    }

    //
    // Photon Tracing OptixEntryPoint
    //

    {
        Program generatorProgram
            = m_context->createProgramFromPTXFile(getPtxFile("renderer/ppm/PhotonGenerator.ptx"), "generator");
        Program exceptionProgram
            = m_context->createProgramFromPTXFile(getPtxFile("renderer/ppm/PhotonGenerator.ptx"), "exception");
        Program missProgram
            = m_context->createProgramFromPTXFile(getPtxFile("renderer/ppm/PhotonGenerator.ptx"), "miss");
        m_context->setRayGenerationProgram(OptixEntryPoint::PPM_PHOTON_PASS, generatorProgram);
        m_context->setMissProgram(OptixEntryPoint::PPM_PHOTON_PASS, missProgram);
        m_context->setExceptionProgram(OptixEntryPoint::PPM_PHOTON_PASS, exceptionProgram);
    }

    m_photons = m_context->createBuffer(RT_BUFFER_OUTPUT);
    m_photons->setFormat(RT_FORMAT_USER);
    m_photons->setElementSize(sizeof(Photon));
    m_photons->setSize(NUM_PHOTONS);
    m_context["photons"]->set(m_photons);
    m_context["photonsSize"]->setUint(NUM_PHOTONS);

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH

    optix::Buffer photonsHashTableCount
        = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, NUM_PHOTONS);
    m_context["photonsHashTableCount"]->set(photonsHashTableCount);
    {
        Program program
            = m_context->createProgramFromPTXFile(getPtxFile("renderer/ppm/UniformGridPhotonInitialize.ptx"), "kernel");
        m_context->setRayGenerationProgram(OptixEntryPoint::PPM_CLEAR_PHOTONS_UNIFORM_GRID_PASS, program);
    }
    m_context["photonsGridCellSize"]->setFloat(0.0f);
    m_context["photonsGridCellSize"]->setFloat(0.0f);
    m_context["photonsGridSize"]->setUint(0, 0, 0);
    m_context["photonsWorldOrigo"]->setFloat(make_float3(0));

#elif ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_KD_TREE_CPU

    m_photonKdTreeSize = pow2roundup(NUM_PHOTONS + 1) - 1;
    m_photonKdTree = m_context->createBuffer(RT_BUFFER_INPUT);
    m_photonKdTree->setFormat(RT_FORMAT_USER);
    m_photonKdTree->setElementSize(sizeof(Photon));
    m_photonKdTree->setSize(m_photonKdTreeSize);
    m_context["photonKdTree"]->set(m_photonKdTree);

#elif ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_UNIFORM_GRID

    m_context["photonsGridCellSize"]->setFloat(0.0f);
    m_context["photonsGridSize"]->setUint(0, 0, 0);
    m_context["photonsWorldOrigo"]->setFloat(make_float3(0));
    m_photonsHashCells = m_context->createBuffer(RT_BUFFER_OUTPUT);
    m_photonsHashCells->setFormat(RT_FORMAT_UNSIGNED_INT);
    m_photonsHashCells->setSize(NUM_PHOTONS);
    m_hashmapOffsetTable = m_context->createBuffer(RT_BUFFER_OUTPUT);
    m_hashmapOffsetTable->setFormat(RT_FORMAT_UNSIGNED_INT);
    m_hashmapOffsetTable->setSize(PHOTON_GRID_MAX_SIZE + 1);
    m_context["hashmapOffsetTable"]->set(m_hashmapOffsetTable);

#endif

    //
    // Volumetric Photon Spheres buffer
    //
#if ENABLE_PARTICIPATING_MEDIA
    {
        m_volumetricPhotonsBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
        m_volumetricPhotonsBuffer->setFormat(RT_FORMAT_USER);
        m_volumetricPhotonsBuffer->setElementSize(sizeof(Photon));
        m_volumetricPhotonsBuffer->setSize(NUM_VOLUMETRIC_PHOTONS);
        m_context["volumetricPhotons"]->setBuffer(m_volumetricPhotonsBuffer);

        optix::Geometry photonSpheres = m_context->createGeometry();
        photonSpheres->setPrimitiveCount(NUM_VOLUMETRIC_PHOTONS);
        photonSpheres->setIntersectionProgram(
            m_context->createProgramFromPTXFile(getPtxFile("renderer/ppm/VolumetricPhotonSphere.ptx"), "intersect"));
        photonSpheres->setBoundingBoxProgram(
            m_context->createProgramFromPTXFile(getPtxFile("renderer/ppm/VolumetricPhotonSphere.ptx"), "boundingBox"));

        optix::Material material = m_context->createMaterial();
        material->setAnyHitProgram(
            RayType::VOLUMETRIC_RADIANCE,
            m_context->createProgramFromPTXFile(
                getPtxFile("renderer/ppm/VolumetricPhotonSphereRadiance.ptx"), "anyHitRadiance"));
        optix::GeometryInstance volumetricPhotonSpheres
            = m_context->createGeometryInstance(photonSpheres, &material, &material + 1);
        volumetricPhotonSpheres["photonsBuffer"]->setBuffer(m_volumetricPhotonsBuffer);

        m_volumetricPhotonsRoot = m_context->createGeometryGroup();
        m_volumetricPhotonsRoot->setChildCount(1);
        optix::Acceleration m_volumetricPhotonSpheresAcceleration = m_context->createAcceleration("MedianBvh", "Bvh");
        m_volumetricPhotonsRoot->setAcceleration(m_volumetricPhotonSpheresAcceleration);
        m_volumetricPhotonsRoot->setChild(0, volumetricPhotonSpheres);
        m_context["volumetricPhotonsRoot"]->set(m_volumetricPhotonsRoot);
    }

    //
    // Clear Volumetric Photons Program
    //

    {
        Program program
            = m_context->createProgramFromPTXFile(getPtxFile("renderer/ppm/VolumetricPhotonInitialize.ptx"), "kernel");
        m_context->setRayGenerationProgram(OptixEntryPoint::PPM_CLEAR_VOLUMETRIC_PHOTONS_PASS, program);
    }
#endif

    //
    // Indirect Radiance Estimation Buffer
    //

    m_indirectRadianceBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, m_width, m_height);
    m_context["indirectRadianceBuffer"]->set(m_indirectRadianceBuffer);

    //
    // Indirect Radiance Estimation Program
    //
    {
        Program program
            = m_context->createProgramFromPTXFile(getPtxFile("renderer/ppm/IndirectRadianceEstimation.ptx"), "kernel");
        m_context->setRayGenerationProgram(OptixEntryPoint::PPM_INDIRECT_RADIANCE_ESTIMATION_PASS, program);
    }

    //
    // Direct Radiance Estimation Buffer
    //

    m_directRadianceBuffer = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, m_width, m_height);
    m_context["directRadianceBuffer"]->set(m_directRadianceBuffer);

    //
    // Direct Radiance Estimation Program
    //
    {
        Program program
            = m_context->createProgramFromPTXFile(getPtxFile("renderer/ppm/DirectRadianceEstimation.ptx"), "kernel");
        m_context->setRayGenerationProgram(OptixEntryPoint::PPM_DIRECT_RADIANCE_ESTIMATION_PASS, program);
    }

    //
    // Output Buffer
    //
    {
        m_outputBuffer = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, m_width, m_height);
        m_context["outputBuffer"]->set(m_outputBuffer);
    }

    //
    // Output Program
    //
    {
        Program program = m_context->createProgramFromPTXFile(getPtxFile("renderer/ppm/Output.ptx"), "kernel");
        m_context->setRayGenerationProgram(OptixEntryPoint::PPM_OUTPUT_PASS, program);
    }

    //
    // Random state buffer (must be large enough to give states to both photons and image pixels)
    //

    m_randomStatesBuffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL);
    m_randomStatesBuffer->setFormat(RT_FORMAT_USER);
    m_randomStatesBuffer->setElementSize(sizeof(RandomState));
    m_randomStatesBuffer->setSize(PHOTON_LAUNCH_WIDTH, PHOTON_LAUNCH_HEIGHT);
    m_context["randomStates"]->set(m_randomStatesBuffer);

    //
    // Light sources buffer
    //

    m_lightBuffer = m_context->createBuffer(RT_BUFFER_INPUT);
    m_lightBuffer->setFormat(RT_FORMAT_USER);
    m_lightBuffer->setElementSize(sizeof(Light));
    m_lightBuffer->setSize(1);
    m_context["lights"]->set(m_lightBuffer);

    //
    // Debug buffers
    //

    createGpuDebugBuffers();

    m_initialized = true;

    // printf("Num CPU threads: %d\n", m_context->getCPUNumThreads());
    // printf("GPU paging active: %d\n", m_context->getGPUPagingActive());
    // printf("Enabled devices count: %d\n", m_context->getEnabledDeviceCount());
    // printf("Get devices count: %d\n", m_context->getDeviceCount());
    // printf("Used host memory: %d\n", m_context->getUsedHostMemory());
    // printf("Sizeof Photon %d\n", sizeof(Photon));
}

void OptixRenderer::initDevice(const ComputeDevice& device)
{
    // Set OptiX device as given by ComputeDevice::getDeviceId (Cuda ordinal)

    unsigned int deviceCount = m_context->getDeviceCount();
    int deviceOptixOrdinal = -1;
    for (unsigned int index = 0; index < deviceCount; ++index)
    {
        int cudaDeviceOrdinal;
        if (RTresult code
            = rtDeviceGetAttribute(index, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL, sizeof(int), &cudaDeviceOrdinal))
            throw Exception::makeException(code, 0);

        if (cudaDeviceOrdinal == device.getDeviceId())
        {
            deviceOptixOrdinal = index;
        }
    }

    m_optixDeviceOrdinal = deviceOptixOrdinal;

    if (deviceOptixOrdinal >= 0)
    {
        m_context->setDevices(&deviceOptixOrdinal, &deviceOptixOrdinal + 1);
    }
    else
    {
        throw std::runtime_error(
            "Did not find OptiX device Number for given device. OptiX may not support this device.");
    }
}

void OptixRenderer::initScene(IScene& scene)
{
    if (!m_initialized)
    {
        throw std::runtime_error("Cannot initialize scene before OptixRenderer.");
    }

    const QVector<Light>& lights = scene.getSceneLights();
    if (lights.size() == 0)
    {
        throw std::runtime_error("No lights exists in this scene.");
    }

    try
    {
        m_sceneRootGroup = scene.getSceneRootGroup(m_context);
        m_context["sceneRootObject"]->set(m_sceneRootGroup);
        m_sceneAABB = scene.getSceneAABB();
        Sphere sceneBoundingSphere = m_sceneAABB.getBoundingSphere();
        m_context["sceneBoundingSphere"]->setUserData(sizeof(Sphere), &sceneBoundingSphere);

        // Add the lights from the scene to the light buffer

        m_lightBuffer->setSize(lights.size());
        Light* lights_host = (Light*)m_lightBuffer->map();
        memcpy(lights_host, scene.getSceneLights().constData(), sizeof(Light) * lights.size());
        m_lightBuffer->unmap();

        compile();
    }
    catch (const optix::Exception& e)
    {
        QString error = QString("An OptiX error occurred when initializing scene: %1").arg(e.getErrorString().c_str());
        throw std::runtime_error(error.toStdString());
    }
}

void OptixRenderer::compile()
{
    try
    {
        m_context->validate();
        m_context->compile();
    }
    catch (const Exception& e)
    {
        throw e;
    }
}

void OptixRenderer::renderNextIteration(
    unsigned long long iterationNumber,
    unsigned long long localIterationNumber,
    float PPMRadius,
    bool createOutput,
    const RenderServerRenderRequestDetails& details)
{
    // printf("----------------------- %d Local: %d\n", iterationNumber, localIterationNumber);
    if (!m_initialized)
    {
        throw std::runtime_error("Traced before OptixRenderer was initialized.");
    }

    char buffer[40];
    sprintf(buffer, "OptixRenderer::Trace Iteration %lld", iterationNumber);
    ////nvtx::ScopedRange r(buffer);

    try
    {
        // If the width and height of the current render request has changed, we must resize buffers
        if (details.getWidth() != m_width || details.getHeight() != m_height)
        {
            this->resizeBuffers(details.getWidth(), details.getHeight());
        }

        const Camera& camera = details.getCamera();
        const RenderMethod renderMethod = details.getRenderMethod();

        double traceStartTime = currentTime();

        m_context["camera"]->setUserData(sizeof(Camera), &camera);
        m_context["iterationNumber"]->setFloat(static_cast<float>(iterationNumber));
        m_context["localIterationNumber"]->setUint((unsigned int)localIterationNumber);

        if (renderMethod == RenderMethod::PATH_TRACING)
        {
            {
                m_context["ptDirectLightSampling"]->setInt(1);
                // nvtx::ScopedRange r("OptixEntryPoint::PT_RAYTRACE_PASS");
                m_context->launch(
                    OptixEntryPoint::PT_RAYTRACE_PASS,
                    static_cast<unsigned int>(m_width),
                    static_cast<unsigned int>(m_height));
            }
        }
        else
        {

            // Update PPM Radius for next photon tracing pass

            const float ppmAlpha = details.getPPMAlpha();
            m_context["ppmAlpha"]->setFloat(ppmAlpha);
            const float ppmRadiusSquared = PPMRadius * PPMRadius;
            m_context["ppmRadius"]->setFloat(PPMRadius);
            m_context["ppmRadiusSquared"]->setFloat(ppmRadiusSquared);
            const float ppmRadiusSquaredNew
                = ppmRadiusSquared * (iterationNumber + ppmAlpha) / float(iterationNumber + 1);
            m_context["ppmRadiusSquaredNew"]->setFloat(ppmRadiusSquaredNew);

#if ENABLE_PARTICIPATING_MEDIA
            m_context["volumetricRadius"]->setFloat(0.033 / 0.033 * PPMRadius);

            //
            // Clear volume photons
            //

            {
                // nvtx::ScopedRange r( "OptixEntryPoint::PPM_CLEAR_VOLUMETRIC_PHOTONS_PASS" );
                m_context->launch(OptixEntryPoint::PPM_CLEAR_VOLUMETRIC_PHOTONS_PASS, NUM_VOLUMETRIC_PHOTONS);
            }
#endif
            // Set up the uniform grid bounds

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH
            {
                // nvtx::ScopedRange r("initializeStochasticHashPhotonMap()");
                initializeStochasticHashPhotonMap(PPMRadius);
            }
#endif

            //
            // Photon Tracing
            //

            {
                // nvtx::ScopedRange r( "OptixEntryPoint::PHOTON_PASS" );
                m_context->launch(
                    OptixEntryPoint::PPM_PHOTON_PASS,
                    static_cast<unsigned int>(PHOTON_LAUNCH_WIDTH),
                    static_cast<unsigned int>(PHOTON_LAUNCH_HEIGHT));

                float totalEmitted = (iterationNumber + 1) * EMITTED_PHOTONS_PER_ITERATION;
                m_context["totalEmitted"]->setFloat(static_cast<float>(totalEmitted));
            }

            debugOutputPhotonTracing();

            //
            // Create Photon Map
            //
            {
                // nvtx::ScopedRange r( "Creating photon map" );
#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_KD_TREE_CPU
                createPhotonKdTreeOnCPU();
#elif ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_UNIFORM_GRID
                createUniformGridPhotonMap(PPMRadius);
#endif
            }

#if ENABLE_PARTICIPATING_MEDIA
            //
            // Rebuild the volumetric photons BVH
            //
            {
                double to = currentTime();
                m_volumetricPhotonsRoot->getAcceleration()->markDirty();
                m_context->launch(OptixEntryPoint::PPM_RAYTRACE_PASS, 0, 0);
                double t1 = currentTime();
                if (iterationNumber % 20 == 0 && iterationNumber < 100)
                {
                    printf("Rebuilt volumetric photons (%d photons) in %.4f.\n", NUM_VOLUMETRIC_PHOTONS, t1 - t0);
                }
            }
#endif

            //
            // Transfer any data from the photon acceleration structure build to the GPU (trigger an empty launch)
            //
            {
                // nvtx::ScopedRange r("Transfer photon map to GPU");
                m_context->launch(OptixEntryPoint::PPM_INDIRECT_RADIANCE_ESTIMATION_PASS, 0, 0);
            }

            // Trace viewing rays
            {
                // nvtx::ScopedRange r("OptixEntryPoint::RAYTRACE_PASS");
                m_context->launch(
                    OptixEntryPoint::PPM_RAYTRACE_PASS,
                    static_cast<unsigned int>(m_width),
                    static_cast<unsigned int>(m_height));
            }

            //
            // PPM Indirect Estimation (using the photon map)
            //

            {
                // nvtx::ScopedRange r("OptixEntryPoint::INDIRECT_RADIANCE_ESTIMATION");
                m_context->launch(OptixEntryPoint::PPM_INDIRECT_RADIANCE_ESTIMATION_PASS, m_width, m_height);
            }

            //
            // Direct Radiance Estimation
            //

            {
                // nvtx::ScopedRange r("OptixEntryPoint::PPM_DIRECT_RADIANCE_ESTIMATION_PASS");
                m_context->launch(OptixEntryPoint::PPM_DIRECT_RADIANCE_ESTIMATION_PASS, m_width, m_height);
            }

            //
            // Combine indirect and direct buffers in the output buffer
            //

            // nvtx::ScopedRange r("OptixEntryPoint::PPM_OUTPUT_PASS");
            m_context->launch(OptixEntryPoint::PPM_OUTPUT_PASS, m_width, m_height);
        }

        double end = currentTime();
        double traceTime = end - traceStartTime;
    }
    catch (const optix::Exception& e)
    {
        QString error = QString("An OptiX error occurred: %1").arg(e.getErrorString().c_str());
        throw std::runtime_error(error.toStdString());
    }
}

void OptixRenderer::resizeBuffers(unsigned int width, unsigned int height)
{
    m_outputBuffer->setSize(width, height);
    m_raytracePassOutputBuffer->setSize(width, height);
    m_outputBuffer->setSize(width, height);
    m_directRadianceBuffer->setSize(width, height);
    m_indirectRadianceBuffer->setSize(width, height);
    m_randomStatesBuffer->setSize(
        std::max((unsigned int)PHOTON_LAUNCH_WIDTH, 1280U), std::max((unsigned int)PHOTON_LAUNCH_HEIGHT, 768U));
    initializeRandomStates();
    m_width = width;
    m_height = height;
}

unsigned int OptixRenderer::getWidth() const
{
    return m_width;
}

unsigned int OptixRenderer::getHeight() const
{
    return m_height;
}

void OptixRenderer::getOutputBuffer(void* data)
{
    void* buffer = reinterpret_cast<void*>(m_outputBuffer->map());
    memcpy(data, buffer, getScreenBufferSizeBytes());
    m_outputBuffer->unmap();
}

unsigned int OptixRenderer::getScreenBufferSizeBytes() const
{
    return m_width * m_height * sizeof(optix::float3);
}

void OptixRenderer::debugOutputPhotonTracing()
{
#if ENABLE_RENDER_DEBUG_OUTPUT
    printf(
        "Grid size: %d %d %d. Cellsize: %.4f\n",
        m_gridSize.x,
        m_gridSize.y,
        m_gridSize.z,
        m_context["photonsGridCellSize"]->getFloat());
    {
        optix::Buffer buffer = m_context["debugPhotonPathLengthBuffer"]->getBuffer();
        unsigned int* buffer_Host = (unsigned int*)buffer->map();
        unsigned long long sumPaths = 0;
        unsigned int numZero = 0;
        for (int i = 0; i < PHOTON_LAUNCH_WIDTH * PHOTON_LAUNCH_HEIGHT; i++)
        {
            sumPaths += buffer_Host[i];
            if (buffer_Host[i] == 0)
            {
                numZero++;
            }
        }
        buffer->unmap();
        double averagePathLength = double(sumPaths) / (PHOTON_LAUNCH_WIDTH * PHOTON_LAUNCH_HEIGHT);
        double percentageZero = 100 * double(numZero) / (PHOTON_LAUNCH_WIDTH * PHOTON_LAUNCH_HEIGHT);
        printf("  Average photonprd path length: %.4f (Paths with 0: %.4f%%)\n", averagePathLength, percentageZero);
    }

    {
        optix::Buffer buffer = m_context["debugIndirectRadianceCellsVisisted"]->getBuffer();
        unsigned int* buffer_Host = (unsigned int*)buffer->map();
        unsigned long long sumVisited = 0;
        unsigned int numNotZero = 0;
        for (int i = 0; i < m_width * m_height; i++)
        {
            if (buffer_Host[i] > 0)
            {
                sumVisited += buffer_Host[i];
                numNotZero++;
            }
        }
        buffer->unmap();
        double visitedAvg = double(sumVisited) / (numNotZero);
        printf("  Average cells visited during indirect estimation  (per pixel): %.4f\n", visitedAvg);
    }

    {
        optix::Buffer buffer = m_context["debugIndirectRadiancePhotonsVisisted"]->getBuffer();
        unsigned int* buffer_Host = (unsigned int*)buffer->map();
        unsigned long long sumVisited = 0;
        unsigned int numNotZero = 0;
        for (int i = 0; i < m_width * m_height; i++)
        {
            if (buffer_Host[i] > 0)
            {
                sumVisited += buffer_Host[i];
                numNotZero++;
            }
        }
        buffer->unmap();
        double visitedAvg = double(sumVisited) / (numNotZero);
        printf("  Average photons visited during indirect estimation (per pixel): %.4f\n", visitedAvg);
    }

#if ACCELERATION_STRUCTURE == ACCELERATION_STRUCTURE_STOCHASTIC_HASH
    {
        const unsigned int hashTableSize = NUM_PHOTONS;
        optix::Buffer buffer = m_context["photonsHashTableCount"]->getBuffer();
        unsigned int* buffer_Host = (unsigned int*)buffer->map();
        unsigned int numFilled = 0;
        float sumCollisions = 0;
        for (int i = 0; i < hashTableSize; i++)
        {
            if (buffer_Host[i] > 0)
            {
                numFilled++;
                sumCollisions += buffer_Host[i];
            }
        }
        buffer->unmap();
        double fillRate = 100 * double(numFilled) / hashTableSize;
        double averageCollisions = double(sumCollisions) / numFilled;
        printf(
            "  Table size %d Filled: %d fill%%: %.4f\n  Uniform grid collisions (in filled cells): %.4f\n",
            hashTableSize,
            numFilled,
            fillRate,
            averageCollisions);
    }
#endif
#endif
}

void OptixRenderer::createGpuDebugBuffers()
{
#if ENABLE_RENDER_DEBUG_OUTPUT
    optix::Buffer debugPhotonPathLengthBuffer
        = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, PHOTON_LAUNCH_WIDTH, PHOTON_LAUNCH_HEIGHT);
    m_context["debugPhotonPathLengthBuffer"]->setBuffer(debugPhotonPathLengthBuffer);
    optix::Buffer debugIndirectRadianceCellsVisisted
        = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, 2000, 2000);
    m_context["debugIndirectRadianceCellsVisisted"]->setBuffer(debugIndirectRadianceCellsVisisted);
    optix::Buffer debugIndirectRadiancePhotonsVisisted
        = m_context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_INT, 2000, 2000);
    m_context["debugIndirectRadiancePhotonsVisisted"]->setBuffer(debugIndirectRadiancePhotonsVisisted);
#endif
}

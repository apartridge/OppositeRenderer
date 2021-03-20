project(RenderEngine LANGUAGES CXX C)

set(
    SOURCES
    clientserver/RenderResultPacket.cpp
    clientserver/RenderServerRenderRequest.cpp
    clientserver/RenderServerRenderRequestDetails.cpp
    geometry_instance/AABInstance.cpp
    geometry_instance/GeometryInstance.cpp
    geometry_instance/SphereInstance.cpp
    geometry_instance/Transform.cpp
    material/Diffuse.cpp
    material/DiffuseEmitter.cpp
    material/Glass.cpp
    material/Material.cpp
    material/Mirror.cpp
    material/ParticipatingMedium.cpp
    material/Texture.cpp
    math/AAB.cpp
    math/Sphere.cpp
    math/Vector3.cpp
    renderer/Camera.cpp
    renderer/Light.cpp
    renderer/OptixRenderer_CPUKdTree.cpp
    renderer/OptixRenderer.cpp
    scene/Cornell.cpp
    scene/IScene.cpp
    scene/Scene.cpp
    util/Image.cpp
    util/BenchmarkTimer.cpp
    util/imageformats/libtga/tga.c
    util/imageformats/libtga/tgaread.c
    util/imageformats/libtga/tgawrite.c
    ComputeDevice.cpp
    ComputeDeviceRepository.cpp
)

set(
    CUDA_SOURCES
    renderer/OptixRenderer_SpatialHash.cu
)

set(
    PTX_SOURCES
    geometry_instance/AAB.cu
    geometry_instance/Sphere.cu
    geometry_instance/TriangleMesh.cu
    renderer/parallelogram.cu
    renderer/ppm/DirectRadianceEstimation.cu
    renderer/ppm/IndirectRadianceEstimation.cu
    renderer/ppm/Output.cu
    renderer/ppm/PhotonGenerator.cu
    renderer/ppm/RayGeneratorPPM.cu
    renderer/ppm/UniformGridPhotonInitialize.cu
    renderer/ppm/VolumetricPhotonInitialize.cu
    renderer/ppm/VolumetricPhotonSphere.cu
    renderer/ppm/VolumetricPhotonSphereRadiance.cu
    renderer/pt/RayGeneratorPT.cu
    material/Diffuse.cu
    material/DiffuseEmitter.cu
    material/Glass.cu
    material/Mirror.cu
    material/ParticipatingMedium.cu
    material/Texture.cu
)

add_library(RenderEnginePtx OBJECT ${PTX_SOURCES})
set_property(TARGET RenderEnginePtx PROPERTY CUDA_PTX_COMPILATION ON)
target_include_directories(RenderEnginePtx PUBLIC "${OptiX_INCLUDE}")
target_include_directories(RenderEnginePtx PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC Qt5::Widgets optix ${optix_rpath} ${CUDA_LIBRARIES} ${ASSIMP_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(${PROJECT_NAME} PUBLIC "${OptiX_INCLUDE}")

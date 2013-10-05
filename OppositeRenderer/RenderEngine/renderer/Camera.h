#pragma once

#include "render_engine_export_api.h"
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include "math/Vector3.h"

class Camera
{
    //typedef optix::float3 float3;

public:
    enum AspectRatioMode 
    {
        KeepVertical,
        KeepHorizontal,
        KeepNone
    };

#ifdef __CUDACC__
    Camera()
    {

    }
    ~Camera()
    {

    }
#else

    RENDER_ENGINE_EXPORT_API Camera()
    {
        eye =  Vector3(1,1,1);
        lookat = Vector3(0,0,0);
        up = Vector3(0,1,0);
        hfov = 60.0;
        vfov = 60.0;
        aperture = 0;
        aspectRatioMode = KeepVertical;
        setup();
    }

    RENDER_ENGINE_EXPORT_API Camera(Vector3 _eye, Vector3 _lookat)
        : eye(_eye), lookat(_lookat)
    {
        up = Vector3(0,1,0);
        hfov = 60.0;
        vfov = 60.0;
        aperture = 0;
        aspectRatioMode = KeepVertical;
        setup();
    }

    RENDER_ENGINE_EXPORT_API Camera(Vector3 eye, Vector3 lookat, Vector3 up, float hfov=60, float vfov=60, float aperture=0,
        AspectRatioMode arm = KeepVertical);

    RENDER_ENGINE_EXPORT_API void setup();
    RENDER_ENGINE_EXPORT_API void scaleFOV(float);
    RENDER_ENGINE_EXPORT_API void translate(float, float);
    RENDER_ENGINE_EXPORT_API void dolly(float);
    RENDER_ENGINE_EXPORT_API void transform( const optix::Matrix4x4& trans );
    RENDER_ENGINE_EXPORT_API void setAspectRatio(float ratio);
    RENDER_ENGINE_EXPORT_API void setParameters(Vector3 eye_in, Vector3 lookat_in, Vector3 up_in, float hfov_in, float vfov_in, Camera::AspectRatioMode aspectRatioMode_in);

#endif

    optix::float3 eye, lookat, up;
    float hfov, vfov;
    float aperture;
    optix::float3 lookdir, camera_u, camera_v;
    AspectRatioMode aspectRatioMode;
};

class QDataStream;
RENDER_ENGINE_EXPORT_API QDataStream & operator << (QDataStream & out, const Camera & camera);
RENDER_ENGINE_EXPORT_API QDataStream & operator >> (QDataStream & in, Camera & camera);
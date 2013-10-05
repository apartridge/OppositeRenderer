                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
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

#include "Camera.h"
#include <optixu/optixu_matrix_namespace.h>
#include <iostream>
#include <QDataStream>


# ifdef _WIN32
#   include <cfloat>
#   define ISFINITE _finite
# else
#   include <cmath>
#   define ISFINITE std::isfinite
# endif

#ifdef GLUT_FOUND
#  if defined(__APPLE__)
#    include <GLUT/glut.h>
#  else
#    include <GL/glut.h>
#  endif
#else
#  error "You in big trouble without the GLUT."
#endif

using namespace optix;

namespace {

  /*
    Assigns src to dst if 
    src is not inf and nan!
    
    dst = isReal(src) ? src : dst;
  */

  float assignWithCheck( float& dst, const float &src )
  {
    if( ISFINITE( src ) ) {
      dst = src;
    }

    return dst;
  }

  /*
    Assigns src to dst if all src 
    components are neither inf nor nan!
    
    dst = isReal(src) ? src : dst;
  */

  Vector3 assignWithCheck( optix::float3 & dst, const optix::float3 &src )
  {
    if( ISFINITE( src.x ) && ISFINITE( src.y ) && ISFINITE( src.z ) ) {
      dst = src;
    }

    return dst;
  }

  inline float DtoR(float d)
  {
    return d*(static_cast<float>(M_PI)/180.f);
  }


  inline float RtoD(float r)
  {
    return r*(180.f/static_cast<float>(M_PI));
  }


  inline float Clamp(float val, float min, float max)
  {
    return val > min ? (val < max ? val : max) : min;
  }

  Vector3 projectToSphere( float x, float y, float radius )
  {
    x /= radius;
    y /= radius;
    float rad2 = x*x+y*y;
    if(rad2 > 1.0f) {
      float rad = sqrt(rad2);
      x /= rad;
      y /= rad;
      return Vector3( x, y, 0.0f );
    } else {
      float z = sqrt(1-rad2);
      return Vector3( x, y, z );
    }
  }

  Matrix4x4 rotationMatrix( const float3& _to, const float3& _from ) 
  {
    float3 from = normalize( _from );
    float3 to   = normalize( _to );

    float3 v = cross(from, to);
    float  e = dot(from, to);
    if ( e > 1.0f-1.e-9f ) {
      return Matrix4x4::identity();
    } else {
      float h = 1.0f/(1.0f + e);
      float mtx[16];
      mtx[0] = e + h * v.x * v.x;
      mtx[1] = h * v.x * v.y + v.z;
      mtx[2] = h * v.x * v.z - v.y;
      mtx[3] = 0.0f;

      mtx[4] = h * v.x * v.y - v.z;
      mtx[5] = e + h * v.y * v.y;
      mtx[6] = h * v.y * v.z + v.x;
      mtx[7] = 0.0f; 

      mtx[8] = h * v.x * v.z + v.y;
      mtx[9] = h * v.y * v.z - v.x;
      mtx[10] = e + h * v.z * v.z;
      mtx[11] = 0.0f; 
      
      mtx[12] = 0.0f; 
      mtx[13] = 0.0f; 
      mtx[14] = 0.0f; 
      mtx[15] = 1.0f; 

      return Matrix4x4( mtx );
    }
  }

  Matrix4x4 initWithBasis( const float3& u,
                                 const float3& v,
                                 const float3& w,
                                 const float3& t )
  {
    float m[16];
    m[0] = u.x;
    m[1] = v.x;
    m[2] = w.x;
    m[3] = t.x;
    
    m[4] = u.y;
    m[5] = v.y;
    m[6] = w.y;
    m[7] = t.y;

    m[8] = u.z;
    m[9] = v.z;
    m[10] = w.z;
    m[11] = t.z;

    m[12] = 0.0f;
    m[13] = 0.0f;
    m[14] = 0.0f;
    m[15] = 1.0f;

    return Matrix4x4( m );
  }


  inline float det3 (float a, float b, float c,
                     float d, float e, float f,
                     float g, float h, float i)
  { return a*e*i + d*h*c + g*b*f - g*e*c - d*b*i - a*h*f; }


  void printMatrix( const Matrix4x4& m ) 
  {
    std::cerr << m[0] << " " << m[1] << " " << m[2] << " " << m[3] << "\n"
              << m[4] << " " << m[5] << " " << m[6] << " " << m[7] << "\n"
              << m[8] << " " << m[9] << " " << m[10] << " " << m[11] << "\n"
              << m[12] << " " << m[13] << " " << m[14] << " " << m[15] << std::endl; 
  }

#define mm(i,j) m[i*4+j]
  float det4( const Matrix4x4& m )
  {
    float det;
    det  = mm(0,0) * det3(mm(1,1), mm(1,2), mm(1,3),
                          mm(2,1), mm(2,2), mm(2,3),
                          mm(3,1), mm(3,2), mm(3,3));
    det -= mm(0,1) * det3(mm(1,0), mm(1,2), mm(1,3),
                          mm(2,0), mm(2,2), mm(2,3),
                          mm(3,0), mm(3,2), mm(3,3));
    det += mm(0,2) * det3(mm(1,0), mm(1,1), mm(1,3),
                          mm(2,0), mm(2,1), mm(2,3),
                          mm(3,0), mm(3,1), mm(3,3));
    det -= mm(0,3) * det3(mm(1,0), mm(1,1), mm(1,2),
                          mm(2,0), mm(2,1), mm(2,2),
                          mm(3,0), mm(3,1), mm(3,2));
    return det;
  }

  Matrix4x4 inverse( const Matrix4x4& m )
  {
    Matrix4x4 inverse;
    float det = det4( m );

    inverse[0]  =  det3(mm(1,1), mm(1,2), mm(1,3),
                        mm(2,1), mm(2,2), mm(2,3),
                        mm(3,1), mm(3,2), mm(3,3)) / det;
    inverse[1]  = -det3(mm(0,1), mm(0,2), mm(0,3),
                        mm(2,1), mm(2,2), mm(2,3),
                        mm(3,1), mm(3,2), mm(3,3)) / det;
    inverse[2]  =  det3(mm(0,1), mm(0,2), mm(0,3),
                        mm(1,1), mm(1,2), mm(1,3),
                        mm(3,1), mm(3,2), mm(3,3)) / det;
    inverse[3]  = -det3(mm(0,1), mm(0,2), mm(0,3),
                        mm(1,1), mm(1,2), mm(1,3),
                        mm(2,1), mm(2,2), mm(2,3)) / det;

    inverse[4]  = -det3(mm(1,0), mm(1,2), mm(1,3),
                        mm(2,0), mm(2,2), mm(2,3),
                        mm(3,0), mm(3,2), mm(3,3)) / det;
    inverse[5]  =  det3(mm(0,0), mm(0,2), mm(0,3),
                        mm(2,0), mm(2,2), mm(2,3),
                        mm(3,0), mm(3,2), mm(3,3)) / det;
    inverse[6]  = -det3(mm(0,0), mm(0,2), mm(0,3),
                        mm(1,0), mm(1,2), mm(1,3),
                        mm(3,0), mm(3,2), mm(3,3)) / det;
    inverse[7]  =  det3(mm(0,0), mm(0,2), mm(0,3),
                        mm(1,0), mm(1,2), mm(1,3),
                        mm(2,0), mm(2,2), mm(2,3)) / det;

    inverse[8]  =  det3(mm(1,0), mm(1,1), mm(1,3),
                        mm(2,0), mm(2,1), mm(2,3),
                        mm(3,0), mm(3,1), mm(3,3)) / det;
    inverse[9]  = -det3(mm(0,0), mm(0,1), mm(0,3),
                        mm(2,0), mm(2,1), mm(2,3),
                        mm(3,0), mm(3,1), mm(3,3)) / det;
    inverse[10] =  det3(mm(0,0), mm(0,1), mm(0,3),
                        mm(1,0), mm(1,1), mm(1,3),
                        mm(3,0), mm(3,1), mm(3,3)) / det;
    inverse[11] = -det3(mm(0,0), mm(0,1), mm(0,3),
                        mm(1,0), mm(1,1), mm(1,3),
                        mm(2,0), mm(2,1), mm(2,3)) / det;

    inverse[12] = -det3(mm(1,0), mm(1,1), mm(1,2),
                        mm(2,0), mm(2,1), mm(2,2),
                        mm(3,0), mm(3,1), mm(3,2)) / det;
    inverse[13] =  det3(mm(0,0), mm(0,1), mm(0,2),
                        mm(2,0), mm(2,1), mm(2,2),
                        mm(3,0), mm(3,1), mm(3,2)) / det;
    inverse[14] = -det3(mm(0,0), mm(0,1), mm(0,2),
                        mm(1,0), mm(1,1), mm(1,2),
                        mm(3,0), mm(3,1), mm(3,2)) / det;
    inverse[15] =  det3(mm(0,0), mm(0,1), mm(0,2),
                        mm(1,0), mm(1,1), mm(1,2),
                        mm(2,0), mm(2,1), mm(2,2)) / det;

    return inverse;
}

#undef mm
}

Camera::Camera( Vector3 eye,  Vector3 lookat,  Vector3 up, float hfov, float vfov, float aperture, AspectRatioMode arm)
    : eye(eye)
    , lookat(lookat)
    , up(up)
    , hfov(hfov)
    , vfov(vfov)
    , aperture(aperture)
    , aspectRatioMode(arm)
{
    setup();
}

void Camera::setAspectRatio(float ratio)
{
    float realRatio = ratio;

    const float* inputAngle = 0;
    float* outputAngle = 0;
    switch(aspectRatioMode) {
    case Camera::KeepHorizontal:
        inputAngle = &hfov;
        outputAngle = &vfov;
        realRatio = 1.f/ratio;
        break;
    case Camera::KeepVertical:
        inputAngle = &vfov;
        outputAngle = &hfov;
        break;
    default:
        return;
        break;
    }

    *outputAngle = RtoD(2.0f*atanf(realRatio*tanf(DtoR(0.5f*(*inputAngle)))));

    setup();
}

void Camera::setParameters( Vector3 eye_in,  Vector3 lookat_in,  Vector3 up_in, float hfov_in, float vfov_in, Camera::AspectRatioMode aspectRatioMode_in)
{
    eye = eye_in;
    lookat = lookat_in;
    up = up_in;
    hfov = hfov_in;
    vfov = vfov_in;
    aspectRatioMode = aspectRatioMode_in;

    setup();
}

void Camera::setup()
{
    lookdir = assignWithCheck( lookdir, lookat-eye );  // do not normalize lookdir -- implies focal length
    float lookdir_len = length( lookdir );
    up = assignWithCheck( up, normalize(up));
    camera_u = assignWithCheck( camera_u, normalize( cross(lookdir, up) ) );
    camera_v = assignWithCheck( camera_v, normalize( cross(camera_u, lookdir) ) );
    float ulen = lookdir_len * tanf(DtoR(hfov*0.5f));
    camera_u = assignWithCheck( camera_u, camera_u * ulen );
    float vlen = lookdir_len * tanf(DtoR(vfov*0.5f));
    camera_v = assignWithCheck( camera_v, camera_v * vlen );
}

void Camera::scaleFOV(float scale)
{
    const float fov_min = 0.0f;
    const float fov_max = 120.0f;
    float hfov_new = RtoD(2*atanf(scale*tanf(DtoR(hfov*0.5f))));
    hfov_new = Clamp(hfov_new, fov_min, fov_max);
    float vfov_new = RtoD(2*atanf(scale*tanf(DtoR(vfov*0.5f))));
    vfov_new = Clamp(vfov_new, fov_min, fov_max);

    hfov = assignWithCheck( hfov, hfov_new );
    vfov = assignWithCheck( vfov, vfov_new );

    setup();
}

void Camera::translate(float x, float y)
{
    Vector3 trans = camera_u*x + camera_v*y;
    eye = assignWithCheck( eye, eye + trans );
    lookat = assignWithCheck( lookat, lookat + trans );
    setup();
}


// Here scale will move the eye point closer or farther away from the
// lookat point.  If you want an invertable value feed it
// (previous_scale/(previous_scale-1)
void Camera::dolly(float scale)
{
    // Better make sure the scale isn't exactly one.
    if (scale == 1.0f) return;
    float3 d = (lookat - eye) * scale;
    eye  = assignWithCheck( eye, (optix::float3)eye + d );

    setup();
}

void Camera::transform( const Matrix4x4& trans )
{
    float3 cen = lookat;         // TODO: Add logic for various rotation types (eg, flythrough)

    Matrix4x4 frame = initWithBasis( normalize(camera_u),
        normalize(camera_v),
        normalize(-lookdir),
        cen );
    Matrix4x4 frame_inv = inverse( frame );

    Matrix4x4 final_trans = frame * trans * frame_inv;
    float4 up4     = make_float4( up );
    float4 eye4    = make_float4( eye );
    eye4.w         = 1.0f;
    float4 lookat4 = make_float4( lookat );
    lookat4.w      = 1.0f;


    up = assignWithCheck( up, make_float3( final_trans*up4 ) );
    eye = assignWithCheck( eye, make_float3( final_trans*eye4 ) );
    lookat = assignWithCheck( lookat, make_float3( final_trans*lookat4 ) );

    setup();
}

// Serialize and unserialize Camera object

QDataStream & operator <<( QDataStream & out, const Camera & camera )
{
    out << camera.eye.x << camera.eye.y << camera.eye.z << camera.lookat.x << camera.lookat.y << camera.lookat.z 
        << camera.up.x << camera.up.y << camera.up.z << camera.hfov << camera.vfov << camera.aperture;
    return out;
}

QDataStream & operator >> ( QDataStream & in, Camera & camera )
{
    float eye_x, eye_y, eye_z;
    float lookat_x, lookat_y, lookat_z;
    float up_x, up_y, up_z;
    float hfov;
    float vfov;
    float aperture;

    in >> eye_x >> eye_y >> eye_z >> lookat_x >> lookat_y >> lookat_z
       >> up_x >> up_y >> up_z >> hfov >> vfov >> aperture;

    /*printf("eye_x %.2f ", eye_x);
    printf("eye_y %.2f ", eye_y);
    printf("eye_z %.2f\n", eye_z);
    printf("up_x %.2f ", up_x);
    printf("up_y %.2f ", up_y);
    printf("up_z %.2f\n", up_z);*/

    camera = Camera( Vector3(eye_x, eye_y, eye_z),  Vector3(lookat_x, lookat_y, lookat_z),
                      Vector3(up_x, up_y, up_z), hfov, vfov, aperture);

    return in;
}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
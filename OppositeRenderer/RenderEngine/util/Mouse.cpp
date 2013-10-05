
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

#include "Mouse.h"
#include <optixu/optixu_matrix_namespace.h>
#include <iostream>
#include <QMouseEvent>

# ifdef _WIN32
#   include <cfloat>
#   define ISFINITE _finite
# else
#   include <cmath>
#   define ISFINITE std::isfinite
# endif
/*
#ifdef GLUT_FOUND
#  if defined(__APPLE__)
#    include <GLUT/glut.h>
#  else
#    include <GL/glut.h>
#  endif
#else
#  error "You in big trouble without the GLUT."
#endif
*/


#include "renderer/Camera.h"

#define LEFT_BUTTON Qt::LeftButton
#define RIGHT_BUTTON Qt::RightButton
#define MIDDLE_BUTTON Qt::MiddleButton
#define SHIFT_MODIFIER 123
#define STATE_DOWN 1
#define STATE_UP 2
#define STATE_MOTION 2


//-----------------------------------------------------------------------------
// 
// Helpers
//
//-----------------------------------------------------------------------------

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

    float3 assignWithCheck( float3& dst, const float3 &src )
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


    float3 projectToSphere( float x, float y, float radius )
    {
        x /= radius;
        y /= radius;
        float rad2 = x*x+y*y;
        if(rad2 > 1.0f) {
            float rad = sqrt(rad2);
            x /= rad;
            y /= rad;
            return make_float3( x, y, 0.0f );
        } else {
            float z = sqrt(1-rad2);
            return make_float3( x, y, z );
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
    /*
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
    }*/

#undef mm
}


//-----------------------------------------------------------------------------
// 
// Mouse definition 
//
//-----------------------------------------------------------------------------

Mouse::Mouse(Camera& camera, int xres, int yres)
    : camera(camera)
    , xres(xres)
    , yres(yres)
    , fov_speed(2)
    , dolly_speed(5)
    , translate_speed(33)
{



}

void Mouse::handleMouseFunc(int button, int state, int x, int y, int modifier)
{
    switch(state) {
    case STATE_DOWN:
        current_interaction = InteractionState(modifier, button, state);
        call_func(x,y);
        break;
    case STATE_UP:
        break;
    }
}

void Mouse::handleMoveFunc(int x, int y)
{
    current_interaction.state = STATE_MOTION;
    call_func(x,y);
}

void Mouse::handlePassiveMotionFunc(int x, int y)
{

}

void Mouse::handleResize(int new_xres, int new_yres)
{
    if(new_xres > 0 && new_yres > 0)
    {
        xres = new_xres;
        yres = new_yres;
    }
}

void Mouse::call_func(int x, int y)
{
    //printf("Modifier:%d\nButton:%d\n", current_interaction.modifier, current_interaction.button);

    int modifier = current_interaction.modifier;
    int button   = current_interaction.button;
    if (modifier == 0                 && button == LEFT_BUTTON)
        rotate(x, y);
    if (modifier == 0                 && button == MIDDLE_BUTTON)
        translate(x, y);
    if (modifier == SHIFT_MODIFIER && button == RIGHT_BUTTON)
        fov(x, y);
    if (modifier == 0                 && button == RIGHT_BUTTON)
        dolly(x, y);
}

void Mouse::fov(int x, int y)
{
    if(current_interaction.state == STATE_MOTION) {
        float xmotion = (current_interaction.last_x - x)/static_cast<float>(xres);
        float ymotion = (current_interaction.last_y - y)/static_cast<float>(yres);
        float scale;
        if(fabsf(xmotion) > fabsf(ymotion))
            scale = xmotion;
        else
            scale = ymotion;
        scale *= fov_speed;

        if (scale < 0.0f)
            scale = 1.0f/(1.0f-scale);
        else
            scale += 1.0f;

        // Manipulate Camera
        camera.scaleFOV(scale);
    }
    current_interaction.last_x = x;
    current_interaction.last_y = y;
}


void Mouse::translate(int x, int y)
{
    if(current_interaction.state == STATE_MOTION) {
        float xmotion =  float(current_interaction.last_x - x)/xres;
        float ymotion = -float(current_interaction.last_y - y)/yres;
        camera.translate(xmotion*translate_speed, ymotion*translate_speed);
    }
    current_interaction.last_x = x;
    current_interaction.last_y = y;
}


void Mouse::dolly(int x, int y)
{
    if(current_interaction.state == STATE_MOTION) {
        float xmotion = -float(current_interaction.last_x - x)/xres;
        float ymotion = -float(current_interaction.last_y - y)/yres;

        float scale;
        if(fabsf(xmotion) > fabsf(ymotion))
            scale = xmotion;
        else
            scale = ymotion;
        scale *= dolly_speed;

        camera.dolly(scale);
    }
    current_interaction.last_x = x;
    current_interaction.last_y = y;
}

void Mouse::rotate(int x, int y)
{
    //printf("Rotate! Current state %d\n",  current_interaction.state);
    float xpos = 2.0f*static_cast<float>(x)/static_cast<float>(xres) - 1.0f;
    float ypos = 1.0f - 2.0f*static_cast<float>(y)/static_cast<float>(yres);

    if ( current_interaction.state == STATE_DOWN ) {

        current_interaction.rotate_from = projectToSphere( xpos, ypos, 0.8f );

    } else if(current_interaction.state == STATE_MOTION) {

        float3 to( projectToSphere( xpos, ypos, 0.8f ) );
        float3 from( current_interaction.rotate_from );

        Matrix4x4 m = rotationMatrix( to, from);
        current_interaction.rotate_from = to;
        camera.transform( m ); 
    }
    current_interaction.last_x = x;
    current_interaction.last_y = y;

}

void Mouse::track_and_pan(int x, int y)
{
}

void Mouse::track(int x, int y)
{
}

void Mouse::pan(int x, int y)
{
}

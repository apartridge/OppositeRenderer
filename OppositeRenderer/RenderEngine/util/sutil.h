
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

#ifndef __samples_util_sutil_h__
#define __samples_util_sutil_h__

#include <optix.h>

//#define M_PI 3.14159265359

#ifndef SUTILAPI
#  if sutil_EXPORTS /* Set by CMAKE */
#    if defined( _WIN32 ) || defined( _WIN64 )
#      define SUTILAPI __declspec(dllexport) 
#      define SUTILCLASSAPI
#    elif defined( linux ) || defined( __linux ) || defined ( __CYGWIN__ )
#      define SUTILAPI __attribute__ ((visibility ("default")))
#      define SUTILCLASSAPI SUTILAPI
#    elif defined( __APPLE__ ) && defined( __MACH__ )
#      define SUTILAPI __attribute__ ((visibility ("default")))
#      define SUTILCLASSAPI SUTILAPI
#    else
#      error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"
#    endif

#  else /* sutil_EXPORTS */

#    if defined( _WIN32 ) || defined( _WIN64 )
#      define SUTILAPI __declspec(dllimport)
#      define SUTILCLASSAPI
#    elif defined( linux ) || defined( __linux ) || defined ( __CYGWIN__ )
#      define SUTILAPI __attribute__ ((visibility ("default")))
#      define SUTILCLASSAPI SUTILAPI
#    elif defined( __APPLE__ ) && defined( __MACH__ )
#      define SUTILAPI __attribute__ ((visibility ("default")))
#      define SUTILCLASSAPI SUTILAPI
#    else
#      error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"
#    endif

#  endif /* sutil_EXPORTS */
#endif

#ifdef __cplusplus
extern "C" {
#endif

  /************************************
   **
   **    Error checking helpers 
   **
   ***********************************/

  void SUTILAPI sutilReportError(const char* message);
  void SUTILAPI sutilHandleError(RTcontext context, RTresult code, const char* file, int line);
  void SUTILAPI sutilHandleErrorNoExit(RTcontext context, RTresult code, const char* file, int line);
  void SUTILAPI sutilHandleErrorNoContext(RTresult code, const char* file, int line);

/* assumes current scope has Context variable named 'context' */
#define RT_CHECK_ERROR( func )                                     \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      sutilHandleError( context, code, __FILE__, __LINE__ );       \
  } while(0)

/* assumes current scope has Context pointer variable named 'context' */
#define RT_CHECK_ERROR2( func )                                    \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      sutilHandleError( *context, code, __FILE__, __LINE__ );      \
  } while(0)
 
/* assumes current scope has Context variable named 'context' */
#define RT_CHECK_ERROR_RETURN( func )                              \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS ) {                                     \
      sutilHandleErrorNoExit( context, code, __FILE__, __LINE__ ); \
      return code;                                                 \
    }                                                              \
  } while(0)

/* assumes that there is no context, just print to stderr */
#define RT_CHECK_ERROR_NO_CONTEXT( func )                          \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      sutilHandleErrorNoContext(code, __FILE__, __LINE__ );        \
  } while(0)


  /************************************
   **
   **    Directory helpers 
   **
   ***********************************/

  //SUTILAPI const char* sutilSamplesDir(void);
  //SUTILAPI const char* sutilSamplesPtxDir(void);


  ///************************************
  // **
  // **    Display helper functions
  // **
  // ***********************************/

  ///* Must be called before sutilDisplay*InGlutWindow().  This function
  // * may only be called once per application invocation.  Normally this
  // * should be called before your application's parsing of command line
  // * args since glut will modify the given argument list by removing glut-
  // * specific arguments.
  // */
  //RTresult SUTILAPI sutilInitGlut(int* argc, char** argv);

  //RTresult SUTILAPI sutilDisplayBufferInGlutWindow(const char* window_title, RTbuffer buffer);

  //RTresult SUTILAPI sutilDisplayVBOInGlutWindow(const char* window_title, 
  //                                              unsigned int vboId,
  //                                              unsigned int width,
  //                                              unsigned int height,
  //                                              RTcontext context );

  //RTresult SUTILAPI sutilDisplayFilePPM(const char* filename, RTbuffer buffer);


  ///* This can be called to clean up command line arguments.  This should be
  // * called only if not using sutilInitGlut since glutInit handles this
  // * internally.  Currently removes the OS generated command line args on mac
  // * when launching samples via an app.
  // */
  //RTresult SUTILAPI sutilInit(int* argc, char** argv);


  ///* Load the model geometry and materials into an RTgroup
  // *   filename          : location of the ply or obj file
  // *   context           : the context associated with the resulting obj model
  // *   material_override : overrides the materials given in Obj if not NULL
  // *   group             : return param for the obj model 
  // */
  //RTresult SUTILAPI sutilLoadModel( const char* filename,           
  //                                  RTcontext context,
  //                                  RTmaterial material_override,
  //                                  RTgeometrygroup* geometrygroup ); 


  /* Calculate appropriate U,V,W for pinhole_camera shader.
   *   eye          : camera eye position
   *   lookat       : point in scene camera looks at 
   *   up           : up direction 
   *   hfov         : horizontal field of fiew 
   *   aspect_ratio : image aspect ratio (width/height)
   *   U            : [out] U coord for camera shader
   *   V            : [out] V coord for camera shader
   *   W            : [out] W coord for camera shader
   */
  RTresult SUTILAPI sutilCalculateCameraVariables( float eye[3],
                                                   float lookat[3],
                                                   float up[3],
                                                   float hfov,
                                                   float aspect_ratio,
                                                   float U[3],
                                                   float V[3],
                                                   float W[3] );


  /* Get current time in seconds for benchmarking/timing purposes. 
   *   current_time      : return param for the current time in double
   */
  RTresult SUTILAPI sutilCurrentTime( double* current_time );


  ///* Parse the image dimensions command line option, <width>x<height>
  // * used by most samples.
  // *   arg    : pointer to the command line argument to parse
  // *   width  : return param for the desired width of the image; left unmodified
  // *            upon RT_ERROR_UNKNOWN.
  // *   height : return param for the desired height of the image; left unmodified
  // *            upon RT_ERROR_UNKNOWN.
  // *   return : RT_SUCCESS if valid dimensions could be parsed and
  // *            returned; RT_ERROR_UNKNOWN, otherwise.
  // */
  //RTresult SUTILAPI sutilParseImageDimensions( const char *arg,
  //                                             unsigned int *width,
  //                                             unsigned int *height );

  ///* Same as sutilParseImageDimensions, but parses doubles instead of ints */
  //RTresult SUTILAPI sutilParseFloatDimensions( const char *arg,
  //                                             double *width,
  //                                             double *height );

  ///* Prints results to screen using using the following format:
  // * BENCHMARK: <name> | <fps> | <warmup frame count> | <timed frame count> | <total time>
  // */
  //RTresult SUTILAPI sutilPrintBenchmark( const char * name,
  //                                       double total_time,
  //                                       unsigned int warmup_frames,
  //                                       unsigned int timed_frames );

  ///* Call each frame to handle simple benchmarking
  // */
  //RTresult SUTILAPI sutilFrameBenchmark( const char *name,
  //                                       unsigned int warmup_frames,
  //                                       unsigned int timed_frames);

#ifdef __cplusplus
}
#endif

#endif /* __samples_util_sutil_h__ */

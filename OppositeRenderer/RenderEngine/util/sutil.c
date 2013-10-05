
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

#include "sutil.h"
//#include "ImageDisplay.h"
//#include "ModelLoader.h"
//#include <sampleConfig.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    include<windows.h>
#    include<mmsystem.h>
#else /*Apple and Linux both use this */
#    include<sys/time.h>
#    include <unistd.h>
#    include <dirent.h>
#endif
/*
static int dirExists( const char* path )
{
#if defined(_WIN32)
  DWORD attrib = GetFileAttributes( path );
	return (attrib != INVALID_FILE_ATTRIBUTES) && (attrib & FILE_ATTRIBUTE_DIRECTORY);
#else
  DIR* dir=opendir( path );
  if( dir==NULL )
    return 0;
  else {
    closedir(dir);
    return 1;
  }
#endif
}
*/
void sutilReportError(const char* message)
{
  fprintf( stderr, "OptiX Error: %s\n", message );
//#if defined(_WIN32) && defined(RELEASE_PUBLIC)
  {
    char s[2048];
    sprintf( s, "OptiX Error: %s", message );
    MessageBox( 0, s, "OptiX Error", MB_OK|MB_ICONWARNING|MB_SYSTEMMODAL );
  }
//#endif
}
/*
void sutilHandleError(RTcontext context, RTresult code, const char* file, int line)
{
  sutilHandleErrorNoExit( context, code, file, line );
  exit(1);
}

void sutilHandleErrorNoExit(RTcontext context, RTresult code, const char* file, int line)
{
  const char* message;
  char s[2048];
  rtContextGetErrorString(context, code, &message);
  sprintf(s, "%s\n(%s:%d)", message, file, line);
  sutilReportError( s );
}

void sutilHandleErrorNoContext(RTresult code, const char* file, int line)
{
  char s[2048];
  sprintf(s, "Code: %d\n(%s:%d)", code, file, line);
  sutilReportError( s );
  exit(1);
}*/
/*
const char* sutilSamplesDir(void)
{
  const char* dir;
  static char s[512];

  // Allow for overrides.
  dir = getenv( "OPTIX_SAMPLES_DIR" );
  if( dir ) {
    strcpy( s, dir );
    return s;
  }

  // Return hardcoded path if it exists.
  /*if( dirExists(SAMPLES_DIR) )
    return SAMPLES_DIR;
	*/
  // Last resort.*/
  /*return ".";
}*/

//const char* sutilSamplesPtxDir(void)
//{
//  const char* dir;
//  static char s[512];
//
//  // Allow for overrides.
//  dir = getenv( "OPTIX_SAMPLES_PTX_DIR" );
//  if( dir ) {
//    strcpy( s, dir );
//    return s;
//  }
//
//  // Return hardcoded path if it exists.
//  /*if( dirExists(SAMPLES_PTX_DIR) )
//    return SAMPLES_PTX_DIR;*/
//
//  // Last resort.
//  return ".";
//}
/*
RTresult sutilInitGlut(int* argc, char** argv)
{
  return initGlut(argc, argv);
}

RTresult sutilDisplayBufferInGlutWindow(const char* window_title, RTbuffer buffer)
{
  return displayGlutWindow(window_title, buffer);
}

RTresult sutilDisplayVBOInGlutWindow(const char* window_title,
                                   unsigned int vboId,
                                   unsigned int width,
                                   unsigned int height,
                                   RTcontext context)
{
  return displayGlutWindowVBO(window_title, vboId, width, height, context);
}

RTresult sutilDisplayFilePPM(const char* filename, RTbuffer buffer)
{
  return displayFilePPM(filename, buffer);
}

RTresult sutilInit(int* argc, char** argv)
{
  return cleanCommandLineArgs(argc, argv);
}*/
/*
RTresult sutilLoadModel( const char* file, 
                         RTcontext context,
                         RTmaterial material_override,
                         RTgeometrygroup* geometrygroup )
{
  return loadModel( file, context, material_override, geometrygroup );
}
*/
//static void cross( const float a[3], const float b[3], float result[3] )
//{
//  result[0] = a[1]*b[2] - a[2]*b[1];
//  result[1] = a[2]*b[0] - a[0]*b[2];
//  result[2] = a[0]*b[1] - a[1]*b[0];
//}
//
//
//static float dot( const float a[3], const float b[3] )
//{
//  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
//}
//
//
//static void normalize( float a[3] )
//{
//  float inv_len = 1.0f / sqrtf( dot(a, a) );
//  a[0] *= inv_len;
//  a[1] *= inv_len;
//  a[2] *= inv_len;
//}
//
//RTresult sutilCalculateCameraVariables( float eye[3],
//                                        float lookat[3], 
//                                        float up[3],
//                                        float hfov,
//                                        float aspect_ratio,
//                                        float U[3],
//                                        float V[3],
//                                        float W[3] )
//{
//  float ulen, vlen, wlen;
//  W[0] = lookat[0] - eye[0];
//  W[1] = lookat[1] - eye[1];  /* Do not normalize W -- it implies focal length */
//  W[2] = lookat[2] - eye[2];
//
//  wlen = sqrtf( dot( W, W ) );
//  cross( W, up, U );
//  normalize( U );
//  cross( U, W, V );
//  normalize( V );
//  ulen = wlen * tanf( hfov / 2.0f * 3.14159265358979323846f / 180.0f );
//  U[0] *= ulen;
//  U[1] *= ulen;
//  U[2] *= ulen;
//  vlen =  ulen/aspect_ratio;
//  V[0] *= vlen;
//  V[1] *= vlen;
//  V[2] *= vlen;
//  
//  return RT_SUCCESS;
//}

#if defined(_WIN32)

// inv_freq is 1 over the number of ticks per second.
static double inv_freq;
static int freq_initialized = 0;
static int use_high_res_timer = 0;

RTresult sutilCurrentTime( double* current_time )
{
  if(!freq_initialized) {
    LARGE_INTEGER freq;
    use_high_res_timer = QueryPerformanceFrequency(&freq);
    inv_freq = 1.0/freq.QuadPart;
    freq_initialized = 1;
  }
  if (use_high_res_timer) {
    LARGE_INTEGER c_time;
    if(QueryPerformanceCounter(&c_time)) {
      *current_time = c_time.QuadPart*inv_freq;
    } else {
      return RT_ERROR_UNKNOWN;
    }
  } else {
    *current_time = ( (double)timeGetTime() ) * 1.0e-3;
  }
  return RT_SUCCESS;
}

#else

RTresult sutilCurrentTime( double* current_time )
{
  struct timeval tv;
  if( gettimeofday( &tv, 0 ) ) {
    fprintf( stderr, "sutilCurrentTime(): gettimeofday failed!\n" );
    return RT_ERROR_UNKNOWN;
  }

  *current_time = tv.tv_sec+ tv.tv_usec * 1.0e-6;
  return RT_SUCCESS;
}

#endif

//RTresult sutilParseImageDimensions( const char *arg,
//                                    unsigned int *width,
//                                    unsigned int *height )
//{
//  RTresult result = RT_ERROR_UNKNOWN;
//
//  /* look for an 'x': <width>x<height> */
//  size_t width_end = strchr( arg, 'x' ) - arg;
//  size_t height_begin = width_end + 1;
//
//  if ( height_begin < strlen( arg ) ) {
//    /* find the beginning of the height string */
//    const char *height_arg = &arg[height_begin];
//
//    /* copy width to null-terminated string */
//    char width_arg[32];
//    strncpy( width_arg, arg, width_end );
//    width_arg[width_end] = 0;
//
//    /* terminate the width string */
//    width_arg[width_end] = 0;
//
//    *width = atoi( width_arg );
//    *height = atoi( height_arg );
//    result = RT_SUCCESS;
//  }
//
//  return result;
//}
//
//RTresult sutilParseFloatDimensions( const char *arg,
//                                    double *width,
//                                    double *height )
//{
//  RTresult result = RT_ERROR_UNKNOWN;
//
//  /* look for an 'x': <width>x<height> */
//  size_t width_end = strchr( arg, 'x' ) - arg;
//  size_t height_begin = width_end + 1;
//
//  if ( height_begin < strlen( arg ) ) {
//    /* find the beginning of the height string */
//    const char *height_arg = &arg[height_begin];
//
//    /* copy width to null-terminated string */
//    char width_arg[32];
//    strncpy( width_arg, arg, width_end );
//    width_arg[width_end] = 0;
//
//    /* terminate the width string */
//    width_arg[width_end] = 0;
//
//    *width = atof( width_arg );
//    *height = atof( height_arg );
//    result = RT_SUCCESS;
//  }
//
//  return result;
//}
//
//RTresult sutilPrintBenchmark( const char *name,
//                              double total_time,
//                              unsigned int warmup_frames,
//                              unsigned int timed_frames )
//{
//  printf("BENCHMARK: %s | %g fps | %d (warmup) | %d (timed) | %g sec | %g ms/f \n", name,
//            timed_frames / total_time, warmup_frames, timed_frames, total_time, total_time / timed_frames * 1000); 
//  fflush(stdout);
//  return RT_SUCCESS;
//}
//
//
//RTresult sutilFrameBenchmark( const char *name,
//                              unsigned int warmup_frames,
//                              unsigned int timed_frames)
//{
//  /* Handle benchmarking */
//  static double start_frame_time = 0, end_frame_time = 0;
//  static unsigned int timed_frames_done = 0, warmup_frames_done = 0;
//
//  if(warmup_frames_done < warmup_frames) {
//    warmup_frames_done++;
//  } else if(warmup_frames_done == warmup_frames) {
//    sutilCurrentTime(&start_frame_time);
//    warmup_frames_done++;
//  }else {
//    timed_frames_done++;
//    if(timed_frames_done == timed_frames) {
//      sutilCurrentTime(&end_frame_time);
//      sutilPrintBenchmark( name, end_frame_time - start_frame_time, warmup_frames, timed_frames );
//      exit(0);
//    }
//  }
//
//  return RT_SUCCESS;
//}

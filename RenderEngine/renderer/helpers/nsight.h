// Copyright NVIDIA Corporation 2012
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma error "DO NOT USE"

#pragma once

// Helper for profiling with NVIDIA Nsight tools.
//
// If Nsight is installed, find nvToolsExt headers and libs path in
// environment variable NVTOOLSEXT_PATH. We expect that NVTX_AVAILABLE is defined then.

// Define NVTX_ENABLE before including this header to enable nvToolsExt markers and ranges.

#define NVTX_ENABLE
#define NVTX_AVAILABLE

#ifndef NVTX_AVAILABLE
#ifdef NVTX_ENABLE
#undef NVTX_ENABLE
#endif
#endif

#ifdef _WIN32
#include <stdint.h>
// Control inclusion of stdint.h in nvToolsExt.h
#define NVTX_STDINT_TYPES_ALREADY_DEFINED

#else

#include <stdint.h>
// Control inclusion of stdint.h in nvToolsExt.h
#define NVTX_STDINT_TYPES_ALREADY_DEFINED

#endif //_WIN32

#ifdef NVTX_ENABLE

#include "nvToolsExt.h"

#ifdef _WIN64
#pragma comment(lib, "nvToolsExt64_1.lib")
#else
#pragma comment(lib, "nvToolsExt32_1.lib")
#endif

#define NVTX_MarkEx nvtxMarkEx
#define NVTX_MarkA nvtxMarkA
#define NVTX_MarkW nvtxMarkW
#define NVTX_RangeStartEx nvtxRangeStartEx
#define NVTX_RangeStartA nvtxRangeStartA
#define NVTX_RangeStartW nvtxRangeStartW
#define NVTX_RangeEnd nvtxRangeEnd
#define NVTX_RangePushEx nvtxRangePushEx
#define NVTX_RangePushA nvtxRangePushA
#define NVTX_RangePushW nvtxRangePushW
#define NVTX_RangePop nvtxRangePop
#define NVTX_NameOsThreadA nvtxNameOsThreadA
#define NVTX_NameOsThreadW nvtxNameOsThreadW

#else

struct nvtxEventAttributes_t
{
};
typedef uint64_t nvtxRangeId_t;

#ifndef _MSC_VER
#define __noop(...)
#endif

#define NVTX_MarkEx __noop
#define NVTX_MarkA __noop
#define NVTX_MarkW __noop
#define NVTX_RangeStartEx __noop
#define NVTX_RangeStartA __noop
#define NVTX_RangeStartW __noop
#define NVTX_RangeEnd __noop
#define NVTX_RangePushEx __noop
#define NVTX_RangePushA __noop
#define NVTX_RangePushW __noop
#define NVTX_RangePop __noop
#define NVTX_NameOsThreadA __noop
#define NVTX_NameOsThreadW __noop

#endif

// C++ function templates to enable NvToolsExt functions
namespace nvtx
{
#ifdef NVTX_ENABLE

class Attributes
{
public:
    Attributes()
    {
        clear();
    }
    Attributes& category(uint32_t category)
    {
        m_event.category = category;
        return *this;
    }
    Attributes& color(uint32_t argb)
    {
        m_event.colorType = NVTX_COLOR_ARGB;
        m_event.color = argb;
        return *this;
    }
    Attributes& payload(uint64_t value)
    {
        m_event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
        m_event.payload.ullValue = value;
        return *this;
    }
    Attributes& payload(int64_t value)
    {
        m_event.payloadType = NVTX_PAYLOAD_TYPE_INT64;
        m_event.payload.llValue = value;
        return *this;
    }
    Attributes& payload(double value)
    {
        m_event.payloadType = NVTX_PAYLOAD_TYPE_DOUBLE;
        m_event.payload.dValue = value;
        return *this;
    }
    Attributes& message(const char* message)
    {
        m_event.messageType = NVTX_MESSAGE_TYPE_ASCII;
        m_event.message.ascii = message;
        return *this;
    }
    Attributes& message(const wchar_t* message)
    {
        m_event.messageType = NVTX_MESSAGE_TYPE_UNICODE;
        m_event.message.unicode = message;
        return *this;
    }
    Attributes& clear()
    {
        memset(&m_event, 0, NVTX_EVENT_ATTRIB_STRUCT_SIZE);
        m_event.version = NVTX_VERSION;
        m_event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        return *this;
    }
    const nvtxEventAttributes_t* out() const
    {
        return &m_event;
    }

private:
    nvtxEventAttributes_t m_event;
};

class ScopedRange
{
public:
    ScopedRange(const char* message)
    {
        nvtxRangePushA(message);
    }
    ScopedRange(const wchar_t* message)
    {
        nvtxRangePushW(message);
    }
    ScopedRange(const nvtxEventAttributes_t* attributes)
    {
        nvtxRangePushEx(attributes);
    }
    ScopedRange(const nvtx::Attributes& attributes)
    {
        nvtxRangePushEx(attributes.out());
    }
    ~ScopedRange()
    {
        nvtxRangePop();
    }
};

inline void Mark(const nvtx::Attributes& attrib)
{
    nvtxMarkEx(attrib.out());
}
inline void Mark(const nvtxEventAttributes_t* eventAttrib)
{
    nvtxMarkEx(eventAttrib);
}
inline void Mark(const char* message)
{
    nvtxMarkA(message);
}
inline void Mark(const wchar_t* message)
{
    nvtxMarkW(message);
}
inline nvtxRangeId_t RangeStart(const nvtx::Attributes& attrib)
{
    return nvtxRangeStartEx(attrib.out());
}
inline nvtxRangeId_t RangeStart(const nvtxEventAttributes_t* eventAttrib)
{
    return nvtxRangeStartEx(eventAttrib);
}
inline nvtxRangeId_t RangeStart(const char* message)
{
    return nvtxRangeStartA(message);
}
inline nvtxRangeId_t RangeStart(const wchar_t* message)
{
    return nvtxRangeStartW(message);
}
inline void RangeEnd(nvtxRangeId_t id)
{
    nvtxRangeEnd(id);
}
inline int RangePush(const nvtx::Attributes& attrib)
{
    return nvtxRangePushEx(attrib.out());
}
inline int RangePush(const nvtxEventAttributes_t* eventAttrib)
{
    return nvtxRangePushEx(eventAttrib);
}
inline int RangePush(const char* message)
{
    return nvtxRangePushA(message);
}
inline int RangePush(const wchar_t* message)
{
    return nvtxRangePushW(message);
}
inline void RangePop()
{
    nvtxRangePop();
}
inline void NameCategory(uint32_t category, const char* name)
{
    nvtxNameCategoryA(category, name);
}
inline void NameCategory(uint32_t category, const wchar_t* name)
{
    nvtxNameCategoryW(category, name);
}
inline void NameOsThread(uint32_t threadId, const char* name)
{
    nvtxNameOsThreadA(threadId, name);
}
inline void NameOsThread(uint32_t threadId, const wchar_t* name)
{
    nvtxNameOsThreadW(threadId, name);
}
inline void NameCurrentThread(const char* name)
{
    nvtxNameOsThreadA(::GetCurrentThreadId(), name);
}
inline void NameCurrentThread(const wchar_t* name)
{
    nvtxNameOsThreadW(::GetCurrentThreadId(), name);
}

#else

class Attributes
{
public:
    Attributes()
    {
    }
    Attributes& category(uint32_t category)
    {
        return *this;
    }
    Attributes& color(uint32_t argb)
    {
        return *this;
    }
    Attributes& payload(uint64_t value)
    {
        return *this;
    }
    Attributes& payload(int64_t value)
    {
        return *this;
    }
    Attributes& payload(double value)
    {
        return *this;
    }
    Attributes& message(const char* message)
    {
        return *this;
    }
    Attributes& message(const wchar_t* message)
    {
        return *this;
    }
    Attributes& clear()
    {
        return *this;
    }
    const nvtxEventAttributes_t* out()
    {
        return 0;
    }
};

class ScopedRange
{
public:
    ScopedRange(const char* message)
    {
        (void)message;
    }
    ScopedRange(const wchar_t* message)
    {
        (void)message;
    }
    ScopedRange(const nvtxEventAttributes_t* attributes)
    {
        (void)attributes;
    }
    ScopedRange(const Attributes& attributes)
    {
        (void)attributes;
    }
    ~ScopedRange()
    {
    }
};

inline void Mark(const nvtx::Attributes& attrib)
{
    (void)attrib;
}
inline void Mark(const nvtxEventAttributes_t* eventAttrib)
{
    (void)eventAttrib;
}
inline void Mark(const char* message)
{
    (void)message;
}
inline void Mark(const wchar_t* message)
{
    (void)message;
}
inline nvtxRangeId_t RangeStart(const nvtx::Attributes& attrib)
{
    (void)attrib;
    return 0;
}
inline nvtxRangeId_t RangeStart(const nvtxEventAttributes_t* eventAttrib)
{
    (void)eventAttrib;
    return 0;
}
inline nvtxRangeId_t RangeStart(const char* message)
{
    (void)message;
    return 0;
}
inline nvtxRangeId_t RangeStart(const wchar_t* message)
{
    (void)message;
    return 0;
}
inline void RangeEnd(nvtxRangeId_t id)
{
    (void)id;
}
inline int RangePush(const nvtx::Attributes& attrib)
{
    (void)attrib;
    return -1;
}
inline int RangePush(const nvtxEventAttributes_t* eventAttrib)
{
    (void)eventAttrib;
    return -1;
}
inline int RangePush(const char* message)
{
    (void)message;
    return -1;
}
inline int RangePush(const wchar_t* message)
{
    (void)message;
    return -1;
}
inline int RangePop()
{
    return -1;
}
inline void NameCategory(uint32_t category, const char* name)
{
    (void)category;
    (void)name;
}
inline void NameCategory(uint32_t category, const wchar_t* name)
{
    (void)category;
    (void)name;
}
inline void NameOsThread(uint32_t threadId, const char* name)
{
    (void)threadId;
    (void)name;
}
inline void NameOsThread(uint32_t threadId, const wchar_t* name)
{
    (void)threadId;
    (void)name;
}
inline void NameCurrentThread(const char* name)
{
    (void)name;
}
inline void NameCurrentThread(const wchar_t* name)
{
    (void)name;
}

#endif
} // nvtx

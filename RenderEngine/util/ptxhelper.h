#pragma once
#include <string>
inline std::string getPtxFile(const std::string& file)
{
    return std::string(
        "/home/stian/Projects/OppositeRenderer/build/RenderEngine/CMakeFiles/RenderEnginePtx.dir/" + file);
}
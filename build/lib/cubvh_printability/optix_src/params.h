#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include <cstdint>
struct TraceParams {
    OptixTraversableHandle handle;
    const float3* origins;
    const float3* directions;
    const int32_t* source_faces;
    int64_t* faces;
    float* distances;
    float t_min, t_max;
    const float3* step_directions;
    float epsilon;
    int mode;
};

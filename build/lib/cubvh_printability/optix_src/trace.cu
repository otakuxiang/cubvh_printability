#include "params.h"
#include <optix_device.h>
#include <math_constants.h>
extern "C" { __constant__ TraceParams params; }
extern "C" __global__ void __raygen__trace() {
    unsigned int i = optixGetLaunchIndex().x;
    if (params.mode == 1) {
        // One thread per (point, parity direction). Keep closest-hit stepping
        // and the 256-crossing limit identical to the Python reference, but
        // avoid a host round trip and compaction after every surface crossing.
        const unsigned int direction_id = i % 3;
        float3 origin = params.origins[i / 3];
        const float3 direction = params.directions[direction_id];
        const float3 step = params.step_directions[direction_id];
        unsigned int source = 0xffffffffu;
        int parity = 0;
        for (int crossing = 0; crossing < 256; ++crossing) {
            unsigned int face = 0xffffffffu, distance = __float_as_uint(CUDART_INF_F);
            optixTrace(params.handle, origin, direction, 0.0f, 8.0f, 0.0f,
                       OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
                       0, 1, 0, source, face, distance);
            if (face == 0xffffffffu) {
                params.faces[i] = parity;
                return;
            }
            parity ^= 1;
            // PyTorch evaluates add/multiply/add as separate float32 kernels.
            // Explicit rounded operations prevent fused multiply-add drift.
            const float offset = __fadd_rn(__uint_as_float(distance), params.epsilon);
            origin.x = __fadd_rn(origin.x, __fmul_rn(offset, step.x));
            origin.y = __fadd_rn(origin.y, __fmul_rn(offset, step.y));
            origin.z = __fadd_rn(origin.z, __fmul_rn(offset, step.z));
            source = face;
        }
        params.faces[i] = 2 + parity;  // Unknown, retaining reference parity.
        return;
    }
    unsigned int source = static_cast<unsigned int>(params.source_faces[i]);
    unsigned int face = 0xffffffffu, distance = __float_as_uint(CUDART_INF_F);
    optixTrace(params.handle, params.origins[i], params.directions[i],
               params.t_min, params.t_max, 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE, 0, 1, 0, source, face, distance);
    params.faces[i] = face == 0xffffffffu ? -1 : static_cast<int64_t>(face);
    params.distances[i] = __uint_as_float(distance);
}
extern "C" __global__ void __anyhit__filter() {
    if (optixGetPrimitiveIndex() == optixGetPayload_0()) optixIgnoreIntersection();
}
extern "C" __global__ void __closesthit__trace() {
    optixSetPayload_1(optixGetPrimitiveIndex());
    optixSetPayload_2(__float_as_uint(optixGetRayTmax()));
}
extern "C" __global__ void __miss__trace() {}

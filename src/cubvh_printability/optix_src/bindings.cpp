#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <mutex>
#include "params.h"

#define OX(call) do { auto r = (call); TORCH_CHECK(r == OPTIX_SUCCESS, #call, ": ", optixGetErrorName(r)); } while(0)
#define CU(call) do { auto r = (call); TORCH_CHECK(r == cudaSuccess, #call, ": ", cudaGetErrorString(r)); } while(0)
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };

class OptixRayTracer {
    int device_;
    OptixDeviceContext context_ = nullptr;
    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    std::array<OptixProgramGroup, 3> groups_{};
    OptixTraversableHandle handle_ = 0;
    OptixShaderBindingTable sbt_{};
    torch::Tensor vertices_, faces_, gas_, records_;
    int64_t face_count_;

    void cleanup() noexcept {
        if (pipeline_) optixPipelineDestroy(pipeline_);
        for (auto group : groups_) if (group) optixProgramGroupDestroy(group);
        if (module_) optixModuleDestroy(module_);
        if (context_) optixDeviceContextDestroy(context_);
    }
    void check(const torch::Tensor& t, at::ScalarType dtype) const {
        TORCH_CHECK(t.is_cuda() && t.get_device() == device_ && t.is_contiguous() && t.scalar_type() == dtype,
                    "expected contiguous CUDA tensor on tracer device with correct dtype");
    }
public:
    OptixRayTracer(torch::Tensor vertices, torch::Tensor faces, const std::string& ptx)
      : device_(vertices.is_cuda() ? vertices.get_device() : 0), vertices_(vertices), faces_(faces), face_count_(faces.size(0)) {
        check(vertices, at::kFloat); check(faces, at::kInt);
        TORCH_CHECK(vertices.dim() == 2 && vertices.size(1) == 3 && vertices.size(0) > 0 && vertices.size(0) <= INT_MAX, "invalid vertices");
        TORCH_CHECK(faces.dim() == 2 && faces.size(1) == 3 && face_count_ > 0 && face_count_ <= INT_MAX, "invalid faces");
        c10::cuda::CUDAGuard guard(vertices.device());
        auto stream = at::cuda::getCurrentCUDAStream(device_).stream();
        try {
            static std::once_flag initialized;
            std::call_once(initialized, [] { OX(optixInit()); });
            OX(optixDeviceContextCreate(nullptr, nullptr, &context_));
            OptixModuleCompileOptions mopts{};
            mopts.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            OptixPipelineCompileOptions popts{};
            popts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            popts.numPayloadValues = 3;
            popts.numAttributeValues = 2;
            popts.pipelineLaunchParamsVariableName = "params";
            popts.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
            char log[8192]; size_t log_size = sizeof(log);
#if OPTIX_VERSION >= 70700
            auto status = optixModuleCreate(context_, &mopts, &popts, ptx.data(), ptx.size(), log, &log_size, &module_);
#else
            auto status = optixModuleCreateFromPTX(context_, &mopts, &popts, ptx.data(), ptx.size(), log, &log_size, &module_);
#endif
            TORCH_CHECK(status == OPTIX_SUCCESS, "OptiX module creation failed: ", std::string(log, std::min(log_size, sizeof(log))));
            OptixProgramGroupOptions pgopts{};
            std::array<OptixProgramGroupDesc, 3> desc{};
            desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            desc[0].raygen.module = module_; desc[0].raygen.entryFunctionName = "__raygen__trace";
            desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            desc[1].miss.module = module_; desc[1].miss.entryFunctionName = "__miss__trace";
            desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            desc[2].hitgroup.moduleCH = module_; desc[2].hitgroup.entryFunctionNameCH = "__closesthit__trace";
            desc[2].hitgroup.moduleAH = module_; desc[2].hitgroup.entryFunctionNameAH = "__anyhit__filter";
            log_size = sizeof(log);
            OX(optixProgramGroupCreate(context_, desc.data(), 3, &pgopts, log, &log_size, groups_.data()));
            OptixPipelineLinkOptions lopts{};
            lopts.maxTraceDepth = 1;
            log_size = sizeof(log);
            OX(optixPipelineCreate(context_, &popts, &lopts, groups_.data(), 3, log, &log_size, &pipeline_));
            OptixStackSizes sizes{};
            for (auto group : groups_) {
#if OPTIX_VERSION >= 70700
                OX(optixUtilAccumulateStackSizes(group, &sizes, pipeline_));
#else
                OX(optixUtilAccumulateStackSizes(group, &sizes));
#endif
            }
            unsigned int direct_traversal, direct_state, continuation;
            OX(optixUtilComputeStackSizes(&sizes, 1, 0, 0, &direct_traversal, &direct_state, &continuation));
            OX(optixPipelineSetStackSize(pipeline_, direct_traversal, direct_state, continuation, 1));
            std::array<Record, 3> host_records{};
            for (int i = 0; i < 3; ++i) OX(optixSbtRecordPackHeader(groups_[i], &host_records[i]));
            auto bytes = vertices.options().dtype(at::kByte);
            records_ = torch::empty({static_cast<int64_t>(sizeof(host_records))}, bytes);
            CU(cudaMemcpyAsync(records_.data_ptr(), host_records.data(), sizeof(host_records), cudaMemcpyHostToDevice, stream));
            auto base = reinterpret_cast<CUdeviceptr>(records_.data_ptr());
            sbt_.raygenRecord = base;
            sbt_.missRecordBase = base + sizeof(Record); sbt_.missRecordStrideInBytes = sizeof(Record); sbt_.missRecordCount = 1;
            sbt_.hitgroupRecordBase = base + 2 * sizeof(Record); sbt_.hitgroupRecordStrideInBytes = sizeof(Record); sbt_.hitgroupRecordCount = 1;
            CUdeviceptr vertex_ptr = reinterpret_cast<CUdeviceptr>(vertices_.data_ptr());
            unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
            OptixBuildInput input{};
            input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            input.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
            input.triangleArray.numVertices = vertices_.size(0);
            input.triangleArray.vertexBuffers = &vertex_ptr;
            input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            input.triangleArray.indexStrideInBytes = 3 * sizeof(int);
            input.triangleArray.numIndexTriplets = faces_.size(0);
            input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(faces_.data_ptr());
            input.triangleArray.flags = &flags;
            input.triangleArray.numSbtRecords = 1;
            OptixAccelBuildOptions aopts{};
            aopts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            aopts.operation = OPTIX_BUILD_OPERATION_BUILD;
            OptixAccelBufferSizes buffers{};
            OX(optixAccelComputeMemoryUsage(context_, &aopts, &input, 1, &buffers));
            auto temporary = torch::empty({static_cast<int64_t>(buffers.tempSizeInBytes)}, bytes);
            gas_ = torch::empty({static_cast<int64_t>(buffers.outputSizeInBytes)}, bytes);
            OX(optixAccelBuild(context_, stream, &aopts, &input, 1,
                reinterpret_cast<CUdeviceptr>(temporary.data_ptr()), buffers.tempSizeInBytes,
                reinterpret_cast<CUdeviceptr>(gas_.data_ptr()), buffers.outputSizeInBytes, &handle_, nullptr, 0));
            CU(cudaStreamSynchronize(stream));
        } catch (...) { cudaStreamSynchronize(stream); cleanup(); throw; }
    }
    ~OptixRayTracer() { c10::cuda::CUDAGuard guard(vertices_.device()); cleanup(); }
    torch::Tensor contains_closest(torch::Tensor points, torch::Tensor directions,
                                   torch::Tensor step_directions, float epsilon) {
        check(points, at::kFloat); check(directions, at::kFloat); check(step_directions, at::kFloat);
        TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must be [N,3]");
        TORCH_CHECK(directions.dim() == 2 && directions.size(0) == 3 && directions.size(1) == 3
                    && step_directions.sizes() == directions.sizes(), "expected three parity directions");
        TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0, "epsilon must be positive and finite");
        const auto n = points.size(0);
        TORCH_CHECK(n <= INT_MAX / 3, "too many parity points");
        c10::cuda::CUDAGuard guard(vertices_.device());
        auto stream = at::cuda::getCurrentCUDAStream(device_).stream();
        auto result = torch::empty({n, 3}, points.options().dtype(at::kLong));
        if (!n) return result;
        TraceParams params{};
        params.handle = handle_;
        params.origins = reinterpret_cast<const float3*>(points.data_ptr());
        params.directions = reinterpret_cast<const float3*>(directions.data_ptr());
        params.step_directions = reinterpret_cast<const float3*>(step_directions.data_ptr());
        params.faces = result.data_ptr<int64_t>();
        params.epsilon = epsilon;
        params.mode = 1;
        auto buffer = torch::empty({static_cast<int64_t>(sizeof(params))}, points.options().dtype(at::kByte));
        CU(cudaMemcpyAsync(buffer.data_ptr(), &params, sizeof(params), cudaMemcpyHostToDevice, stream));
        OX(optixLaunch(pipeline_, stream, reinterpret_cast<CUdeviceptr>(buffer.data_ptr()), sizeof(params), &sbt_, n * 3, 1, 1));
        CU(cudaStreamSynchronize(stream));
        return result;
    }
    std::tuple<torch::Tensor, torch::Tensor> trace(torch::Tensor origins, torch::Tensor directions,
                                                float t_min, float t_max, torch::Tensor source) {
        check(origins, at::kFloat); check(directions, at::kFloat); check(source, at::kInt);
        TORCH_CHECK(origins.dim() == 2 && origins.size(1) == 3 && directions.sizes() == origins.sizes(), "rays must be [N,3]");
        auto n = origins.size(0);
        TORCH_CHECK(n <= INT_MAX && source.dim() == 1 && source.size(0) == n, "invalid source ids or ray count");
        TORCH_CHECK(t_min >= 0 && t_max > t_min, "invalid ray interval");
        c10::cuda::CUDAGuard guard(vertices_.device());
        auto stream = at::cuda::getCurrentCUDAStream(device_).stream();
        auto ids = torch::empty({n}, origins.options().dtype(at::kLong));
        auto distances = torch::empty({n}, origins.options());
        if (!n) return {ids, distances};
        TraceParams params{handle_, reinterpret_cast<const float3*>(origins.data_ptr()),
            reinterpret_cast<const float3*>(directions.data_ptr()), source.data_ptr<int32_t>(),
            ids.data_ptr<int64_t>(), distances.data_ptr<float>(), t_min, t_max};
        auto buffer = torch::empty({static_cast<int64_t>(sizeof(params))}, origins.options().dtype(at::kByte));
        CU(cudaMemcpyAsync(buffer.data_ptr(), &params, sizeof(params), cudaMemcpyHostToDevice, stream));
        OX(optixLaunch(pipeline_, stream, reinterpret_cast<CUdeviceptr>(buffer.data_ptr()), sizeof(params), &sbt_, n, 1, 1));
        // Makes host launch params and temporary tensor lifetimes explicit and
        // permits later batches on a different current PyTorch stream.
        CU(cudaStreamSynchronize(stream));
        return {ids, distances};
    }
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<OptixRayTracer>(m, "OptixRayTracer")
        .def(pybind11::init<torch::Tensor, torch::Tensor, const std::string&>())
        .def("trace", &OptixRayTracer::trace)
        .def("contains_closest", &OptixRayTracer::contains_closest);
}

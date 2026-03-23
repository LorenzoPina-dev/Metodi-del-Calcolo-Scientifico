// src/gpu/index.ts
export { initGPU, isGPUAvailable, setGPUEnabled, GPU_THRESHOLD,
         gpuMatmul, gpuElementwise, gpuScalarOp, gpuUnary, gpuJacobi,
         getDevice } from "./webgpu_backend.js";

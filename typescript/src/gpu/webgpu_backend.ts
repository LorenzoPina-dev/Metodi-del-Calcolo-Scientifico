// ============================================================
// src/gpu/webgpu_backend.ts
//
// Singleton WebGPU — inizializzazione device, gestione buffer,
// dispatch compute shader e lettura risultati.
//
// Architettura:
//   • GPUDevice singleton (inizializzato una volta)
//   • Buffer pool per riuso: evita allocazioni ripetute
//   • Tutti i dati vengono convertiti da Float64 (JS) → Float32 (GPU)
//     con conseguente riduzione di precisione (~7 cifre significative).
//   • Soglia GPU: n > GPU_THRESHOLD (default 300, configurabile)
//
// ⚠  ATTENZIONE PRECISIONE:
//   WebGPU/WGSL non supporta f64 nativo.
//   Le operazioni GPU usano f32 (single precision).
//   Per calcoli scientifici che richiedono f64, usare WASM o CPU.
//   Il risultato GPU viene convertito in f64 al ritorno.
// ============================================================

import {
    SHADER_MATMUL, SHADER_ELEMENTWISE, SHADER_SCALAR_OP,
    SHADER_UNARY, SHADER_JACOBI, SHADER_REDUCE_SUM
} from "./shaders.js";

// ─── Soglia GPU ───────────────────────────────────────────────────────────────
export const GPU_THRESHOLD = {
    MATMUL      : 90_000,    // n² ≥ 90k  → n ≥ 300
    ELEMENTWISE : 200_000,   // n² ≥ 200k → n ≥ 450
    JACOBI      : 40_000,    // n² ≥ 40k  → n ≥ 200
} as const;

// ─── Stato globale ────────────────────────────────────────────────────────────
let _device: GPUDevice | null = null;
let _adapter: GPUAdapter | null = null;
let _available = false;
let _enabled   = true;   // può essere disabilitato manualmente

// Pipeline cache: evita ricompilazione shader ad ogni chiamata
const _pipelineCache = new Map<string, GPUComputePipeline>();
const _bglCache      = new Map<string, GPUBindGroupLayout>();

// ─── API pubblica ─────────────────────────────────────────────────────────────

/** Abilita / disabilita il backend GPU globalmente. */
export function setGPUEnabled(enabled: boolean): void { _enabled = enabled; }

/** Ritorna true se WebGPU è disponibile e abilitato. */
export function isGPUAvailable(): boolean { return _available && _enabled; }

/** Inizializza WebGPU. Ritorna true se riuscito. */
export async function initGPU(): Promise<boolean> {
    if (_available) return true;
    if (!_enabled)  return false;

    try {
        if (typeof navigator === "undefined" || !navigator.gpu) {
            // Node.js senza WebGPU polyfill
            // Tentiamo il polyfill @webgpu/node se installato
            try {
                const { create } = await import("webgpu" as any);
                const nav = create([]) as unknown as GPU;
                _adapter = await nav.requestAdapter({ powerPreference: "high-performance" });
            } catch {
                return false;
            }
        } else {
            _adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
        }

        if (!_adapter) return false;

        _device = await _adapter.requestDevice({
            label: "numeric-matrix GPU device",
        });
        _device.lost.then((info) => {
            console.warn("[GPU] Device lost:", info.reason, info.message);
            _available = false;
            _device    = null;
        });

        _available = true;
        console.log(`[GPU] WebGPU inizializzato: ${_adapter.info?.description ?? "unknown"}`);
        return true;
    } catch (e) {
        console.warn("[GPU] WebGPU non disponibile:", (e as Error).message);
        return false;
    }
}

export function getDevice(): GPUDevice {
    if (!_device) throw new Error("GPU non inizializzata. Chiama initGPU() prima.");
    return _device;
}

// ─── Buffer utilities ─────────────────────────────────────────────────────────

/** Crea un buffer GPU da Float64Array, convertendo a Float32. */
export function createF32Buffer(
    data: Float64Array | null,
    size: number,
    usage: GPUBufferUsageFlags
): GPUBuffer {
    const d = getDevice();
    const buf = d.createBuffer({ size: size * 4, usage, mappedAtCreation: data !== null });
    if (data !== null) {
        const mapped = new Float32Array(buf.getMappedRange());
        for (let i = 0; i < data.length; i++) mapped[i] = data[i];
        buf.unmap();
    }
    return buf;
}

/** Crea un buffer uniforme da un Uint32Array o Float32Array. */
export function createUniformBuffer(data: ArrayBuffer): GPUBuffer {
    const d = getDevice();
    const buf = d.createBuffer({
        size   : Math.max(data.byteLength, 16),   // minimo 16 byte per alignment
        usage  : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data));
    buf.unmap();
    return buf;
}

/** Legge un buffer GPU e ritorna Float64Array (convertendo da f32). */
export async function readF32Buffer(buf: GPUBuffer, nElems: number): Promise<Float64Array> {
    const d = getDevice();
    const staging = d.createBuffer({
        size  : nElems * 4,
        usage : GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, nElems * 4);
    d.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const src = new Float32Array(staging.getMappedRange());
    const out = new Float64Array(nElems);
    for (let i = 0; i < nElems; i++) out[i] = src[i];
    staging.unmap();
    staging.destroy();
    return out;
}

// ─── Pipeline cache ───────────────────────────────────────────────────────────

export function getOrCreatePipeline(key: string, shaderSource: string): GPUComputePipeline {
    if (_pipelineCache.has(key)) return _pipelineCache.get(key)!;
    const d = getDevice();
    const module = d.createShaderModule({ code: shaderSource });
    const pipeline = d.createComputePipeline({
        layout : "auto",
        compute: { module, entryPoint: "main" },
    });
    _pipelineCache.set(key, pipeline);
    return pipeline;
}

// ─── Dispatch helper ──────────────────────────────────────────────────────────

export function dispatch(
    pipeline    : GPUComputePipeline,
    bindGroup   : GPUBindGroup,
    workgroupsX : number,
    workgroupsY = 1,
    workgroupsZ = 1
): void {
    const d = getDevice();
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.end();
    d.queue.submit([enc.finish()]);
}

// ─── Operazioni GPU di alto livello ──────────────────────────────────────────

/** GPU Matmul: C = A × B  (f32 su GPU). */
export async function gpuMatmul(
    aFlat: Float64Array, bFlat: Float64Array,
    M: number, K: number, N: number
): Promise<Float64Array> {
    const d = getDevice();

    // Uniform buffer: [M, K, N, 0] (4 u32)
    const uData = new Uint32Array([M, K, N, 0]);
    const uBuf  = createUniformBuffer(uData.buffer);

    const aBuf = createF32Buffer(aFlat, M * K,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const bBuf = createF32Buffer(bFlat, K * N,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const cBuf = createF32Buffer(null, M * N,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    const pipeline = getOrCreatePipeline("matmul", SHADER_MATMUL);
    const bindGroup = d.createBindGroup({
        layout : pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uBuf } },
            { binding: 1, resource: { buffer: aBuf } },
            { binding: 2, resource: { buffer: bBuf } },
            { binding: 3, resource: { buffer: cBuf } },
        ],
    });

    const wgX = Math.ceil(N / 16);
    const wgY = Math.ceil(M / 16);
    dispatch(pipeline, bindGroup, wgX, wgY);

    const result = await readF32Buffer(cBuf, M * N);

    uBuf.destroy(); aBuf.destroy(); bBuf.destroy(); cBuf.destroy();
    return result;
}

/** GPU element-wise:  C[i] = A[i] op B[i]  (f32).
 *  op: 0=add, 1=sub, 2=mul, 3=div */
export async function gpuElementwise(
    aFlat: Float64Array, bFlat: Float64Array,
    op: 0 | 1 | 2 | 3
): Promise<Float64Array> {
    const d   = getDevice();
    const len = aFlat.length;

    const uData = new Uint32Array([len, op, 0, 0]);
    const uBuf  = createUniformBuffer(uData.buffer);
    const aBuf  = createF32Buffer(aFlat, len, GPUBufferUsage.STORAGE);
    const bBuf  = createF32Buffer(bFlat, len, GPUBufferUsage.STORAGE);
    const cBuf  = createF32Buffer(null,  len, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    const pipeline  = getOrCreatePipeline("elementwise", SHADER_ELEMENTWISE);
    const bindGroup = d.createBindGroup({
        layout : pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uBuf } },
            { binding: 1, resource: { buffer: aBuf } },
            { binding: 2, resource: { buffer: bBuf } },
            { binding: 3, resource: { buffer: cBuf } },
        ],
    });

    dispatch(pipeline, bindGroup, Math.ceil(len / 256));
    const result = await readF32Buffer(cBuf, len);
    uBuf.destroy(); aBuf.destroy(); bBuf.destroy(); cBuf.destroy();
    return result;
}

/** GPU scalar op: C[i] = A[i] op scalar.
 *  op: 0=add, 1=sub, 2=mul, 3=div, 4=pow */
export async function gpuScalarOp(
    aFlat: Float64Array, scalar: number, op: 0|1|2|3|4
): Promise<Float64Array> {
    const d   = getDevice();
    const len = aFlat.length;

    // Uniform: [len, op, 0, 0, scalar, 0, 0, 0] = 8×4 = 32 byte
    const uRaw = new ArrayBuffer(32);
    new Uint32Array(uRaw, 0, 4).set([len, op, 0, 0]);
    new Float32Array(uRaw, 16, 4).set([scalar, 0, 0, 0]);
    const uBuf = createUniformBuffer(uRaw);

    const aBuf = createF32Buffer(aFlat, len, GPUBufferUsage.STORAGE);
    const cBuf = createF32Buffer(null,  len, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    const pipeline  = getOrCreatePipeline("scalar_op", SHADER_SCALAR_OP);
    const bindGroup = d.createBindGroup({
        layout : pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uBuf } },
            { binding: 1, resource: { buffer: aBuf } },
            { binding: 2, resource: { buffer: cBuf } },
        ],
    });

    dispatch(pipeline, bindGroup, Math.ceil(len / 256));
    const result = await readF32Buffer(cBuf, len);
    uBuf.destroy(); aBuf.destroy(); cBuf.destroy();
    return result;
}

/** GPU unary: C[i] = f(A[i]).
 *  op: 0=abs,1=neg,2=sqrt,3=exp,4=sin,5=cos,6=tan,7=floor,8=ceil,9=round */
export async function gpuUnary(aFlat: Float64Array, op: number): Promise<Float64Array> {
    const d   = getDevice();
    const len = aFlat.length;

    const uData = new Uint32Array([len, op, 0, 0]);
    const uBuf  = createUniformBuffer(uData.buffer);
    const aBuf  = createF32Buffer(aFlat, len, GPUBufferUsage.STORAGE);
    const cBuf  = createF32Buffer(null,  len, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    const pipeline  = getOrCreatePipeline("unary", SHADER_UNARY);
    const bindGroup = d.createBindGroup({
        layout : pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uBuf } },
            { binding: 1, resource: { buffer: aBuf } },
            { binding: 2, resource: { buffer: cBuf } },
        ],
    });

    dispatch(pipeline, bindGroup, Math.ceil(len / 256));
    const result = await readF32Buffer(cBuf, len);
    uBuf.destroy(); aBuf.destroy(); cBuf.destroy();
    return result;
}

/** GPU Jacobi: esegue maxIter iterazioni di Jacobi su GPU.
 *  Ritorna il vettore soluzione x (f64, convertito da f32). */
export async function gpuJacobi(
    aFlat   : Float64Array,
    bFlat   : Float64Array,
    diagInv : Float64Array,
    n       : number,
    tol     : number,
    maxIter : number
): Promise<Float64Array> {
    const d = getDevice();

    // Buffer persistenti (riusati tra le iterazioni)
    const uData = new Uint32Array([n, 0, 0, 0]);
    const uBuf  = createUniformBuffer(uData.buffer);

    const aBuf   = createF32Buffer(aFlat,   n * n, GPUBufferUsage.STORAGE);
    const bBuf   = createF32Buffer(bFlat,   n,     GPUBufferUsage.STORAGE);
    const dBuf   = createF32Buffer(diagInv, n,     GPUBufferUsage.STORAGE);

    // x e xNew si alternano (ping-pong)
    const xBuf   = d.createBuffer({
        size  : n * 4,
        usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const xnBuf  = d.createBuffer({
        size  : n * 4,
        usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // Buffer per |x_new - x_old|: un valore per elemento
    const diffBuf = d.createBuffer({
        size  : n * 4,
        usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Inizializza x a 0
    getDevice().queue.writeBuffer(xBuf, 0, new Float32Array(n).buffer);

    const pipeline = getOrCreatePipeline("jacobi", SHADER_JACOBI);

    // Workgroup: 256 thread, n / 256 workgroup
    const wg = Math.ceil(n / 256);

    for (let iter = 0; iter < maxIter; iter++) {
        const bindGroup = d.createBindGroup({
            layout : pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uBuf   } },
                { binding: 1, resource: { buffer: aBuf   } },
                { binding: 2, resource: { buffer: bBuf   } },
                { binding: 3, resource: { buffer: xBuf   } },  // x old
                { binding: 4, resource: { buffer: dBuf   } },
                { binding: 5, resource: { buffer: xnBuf  } },  // x new
                { binding: 6, resource: { buffer: diffBuf} },
            ],
        });
        dispatch(pipeline, bindGroup, wg);

        // Copia xNew → x (ping-pong)
        const enc = d.createCommandEncoder();
        enc.copyBufferToBuffer(xnBuf, 0, xBuf, 0, n * 4);
        d.queue.submit([enc.finish()]);

        // Controlla convergenza ogni 10 iter (costoso: richiede readback)
        if (iter % 10 === 9 || iter === maxIter - 1) {
            const diffs    = await readF32Buffer(diffBuf, n);
            const xForAbs  = await readF32Buffer(xBuf, n);
            let maxDiff = 0, maxAbsX = 0;
            for (let i = 0; i < n; i++) {
                if (diffs[i] > maxDiff) maxDiff = diffs[i];
                const ax = xForAbs[i] < 0 ? -xForAbs[i] : xForAbs[i];
                if (ax > maxAbsX) maxAbsX = ax;
            }
            const denom = maxAbsX > 1 ? maxAbsX : 1;
            if (maxDiff / denom < tol) break;
        }
    }

    const result = await readF32Buffer(xBuf, n);

    // Cleanup
    uBuf.destroy(); aBuf.destroy(); bBuf.destroy(); dBuf.destroy();
    xBuf.destroy(); xnBuf.destroy(); diffBuf.destroy();

    return result;
}

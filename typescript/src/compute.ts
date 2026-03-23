// ============================================================
// src/compute.ts
//
// Entry point centralizzato per GPU + Worker parallelism.
//
// Gerarchia di esecuzione (Float64M, n grande):
//   GPU  (WebGPU, n > 300, f32)         ← ~10-50× WASM
//   Workers (CPU threads, n > 100)       ← ~N× WASM (N = core count)
//   WASM + SIMD (n > soglia_wasm)        ← ~3-8× TS
//   TypeScript fast-path (fallback)      ← baseline
//
// Uso:
//   import { initCompute } from "./compute.js";
//   await initCompute();     // inizializza GPU + Worker pool
//   // Da qui tutte le operazioni usano automaticamente il percorso ottimale
// ============================================================

import { initWasm }     from "./wasm/wasm_bridge.js";
import { initParallel } from "./parallel/worker_pool.js";
import { initGPU }      from "./gpu/webgpu_backend.js";

export interface ComputeCapabilities {
    wasm    : boolean;
    workers : boolean;
    gpu     : boolean;
    numWorkers: number;
}

let _caps: ComputeCapabilities | null = null;

/**
 * Inizializza tutti i backend di calcolo disponibili.
 * Chiamare una sola volta all'avvio dell'applicazione.
 *
 * @param options.gpu     - Abilita WebGPU (default true). Disabilitare se f32 è insufficiente.
 * @param options.workers - Abilita Worker threads (default true).
 */
export async function initCompute(options?: {
    gpu?    : boolean;
    workers?: boolean;
}): Promise<ComputeCapabilities> {
    if (_caps) return _caps;

    const useGPU     = options?.gpu     ?? true;
    const useWorkers = options?.workers ?? true;

    // 1. WASM (sincrono dopo load asincrono)
    let wasmOk = false;
    try {
        await initWasm();
        wasmOk = true;
    } catch (e) {
        console.warn("[Compute] WASM non disponibile:", (e as Error).message);
    }

    // 2. Worker Pool (parallelismo CPU)
    let workersOk = false;
    let numWorkers = 0;
    if (useWorkers) {
        try {
            await initParallel();
            const { getPoolSync } = await import("./parallel/worker_pool.js");
            const pool = getPoolSync();
            if (pool?.available) {
                workersOk  = true;
                numWorkers = pool.numWorkers;
            }
        } catch (e) {
            console.warn("[Compute] Worker pool non disponibile:", (e as Error).message);
        }
    }

    // 3. GPU (WebGPU, con timeout per evitare blocchi su sistemi senza GPU)
    let gpuOk = false;
    if (useGPU) {
        try {
            const result = await Promise.race([
                initGPU(),
                new Promise<boolean>(r => setTimeout(() => r(false), 3000))
            ]);
            gpuOk = result;
        } catch (e) {
            console.warn("[Compute] GPU non disponibile:", (e as Error).message);
        }
    }

    _caps = { wasm: wasmOk, workers: workersOk, gpu: gpuOk, numWorkers };

    console.log(
        `[Compute] WASM=${wasmOk ? "✓" : "✗"}  ` +
        `Workers=${workersOk ? `✓ (${numWorkers})` : "✗"}  ` +
        `GPU=${gpuOk ? "✓ (WebGPU f32)" : "✗"}`
    );

    return _caps;
}

/** Ritorna le capabilities rilevate (null se initCompute() non è stato chiamato). */
export function getCapabilities(): ComputeCapabilities | null { return _caps; }

// Re-export per convenienza
export { initWasm }     from "./wasm/wasm_bridge.js";
export { initParallel, getPoolSync } from "./parallel/worker_pool.js";
export { initGPU, isGPUAvailable, setGPUEnabled } from "./gpu/webgpu_backend.js";

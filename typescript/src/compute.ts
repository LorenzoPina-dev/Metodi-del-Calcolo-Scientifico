import { initWasm } from "./wasm/wasm_bridge.js";
import { WasmWorkerPool } from "./parallel/wasm_worker_pool.js";
import { initGPU } from "./gpu/webgpu_backend.js";

export interface ComputeCapabilities {
    wasm: boolean;
    workers: boolean;
    gpu: boolean;
    numWorkers: number;
}

let _caps: ComputeCapabilities | null = null;
let _initPromise: Promise<ComputeCapabilities> | null = null;

export async function initCompute(options?: {
    gpu?: boolean;
    workers?: boolean;
}): Promise<ComputeCapabilities> {

    if (_caps) return _caps;
    if (_initPromise) return _initPromise;

    _initPromise = (async () => {

        const useGPU     = options?.gpu     ?? true;
        const useWorkers = options?.workers ?? true;

        // Avvio PARALLELO di tutti i sottosistemi
        const wasmP = initWasm()
            .then(() => true)
            .catch(() => {
                console.warn("[Compute] WASM non disponibile");
                return false;
            });

        const workersP = (async () => {
            if (!useWorkers) return { ok: false, n: 0 };
            try {
                // WasmWorkerPool sostituisce WorkerPool:
                // gestisce internamente la soglia WASM single-thread / multi-thread.
                const pool = new WasmWorkerPool();
                await pool.init();
                WasmWorkerPool.instance = pool;
                return {
                    ok: true,
                    n: (pool as any).workers?.length ?? 0
                };
            } catch (e) {
                console.warn("[Compute] WasmWorkerPool errore:", (e as Error).message);
                return { ok: false, n: 0 };
            }
        })();

        const gpuP = (async () => {
            if (!useGPU) return false;
            try {
                const timeout = new Promise<boolean>(res =>
                    setTimeout(() => res(false), 2000)
                );
                return await Promise.race([initGPU(), timeout]);
            } catch {
                return false;
            }
        })();

        const [wasmOk, workersRes, gpuOk] = await Promise.all([
            wasmP,
            workersP,
            gpuP
        ]);

        const threshold = WasmWorkerPool.threshold;
        const side      = Math.round(Math.sqrt(threshold));

        const caps: ComputeCapabilities = {
            wasm:       wasmOk,
            workers:    workersRes.ok,
            gpu:        gpuOk,
            numWorkers: workersRes.n
        };

        console.log(
            `[Compute] WASM=${wasmOk ? "✓" : "✗"}  ` +
            `WasmWorkers=${workersRes.ok ? `✓ (${workersRes.n}, soglia=${side}×${side})` : "✗"}  ` +
            `GPU=${gpuOk ? "✓" : "✗"}`
        );

        _caps = caps;
        return caps;

    })();

    return _initPromise;
}

export function getCapabilities(): ComputeCapabilities | null {
    return _caps;
}

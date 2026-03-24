import { initWasm } from "./wasm/wasm_bridge.js";
import { WorkerPool } from "./parallel/worker_pool.js";
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

        const useGPU = options?.gpu ?? true;
        const useWorkers = options?.workers ?? true;

        // 🔹 Avvio PARALLELO (molto importante)
        const wasmP = initWasm()
            .then(() => true)
            .catch(() => {
                console.warn("[Compute] WASM non disponibile");
                return false;
            });

        const workersP = (async () => {
            if (!useWorkers) return { ok: false, n: 0 };

            try {
                const pool = new WorkerPool();
                await pool.init();
                // Salva l'istanza globale accessibile da test/altri moduli
                WorkerPool.instance = pool;

                return {
                    ok: true,
                    n: (pool as any).workers?.length ?? 0
                };
            } catch (e) {
                console.warn("[Compute] Worker pool errore:", (e as Error).message);
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

        // 🔹 sync finale
        const [wasmOk, workersRes, gpuOk] = await Promise.all([
            wasmP,
            workersP,
            gpuP
        ]);

        const caps: ComputeCapabilities = {
            wasm: wasmOk,
            workers: workersRes.ok,
            gpu: gpuOk,
            numWorkers: workersRes.n
        };

        console.log(
            `[Compute] WASM=${wasmOk ? "✓" : "✗"}  ` +
            `Workers=${workersRes.ok ? `✓ (${workersRes.n})` : "✗"}  ` +
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

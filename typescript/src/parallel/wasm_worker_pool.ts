// ============================================================
// src/parallel/wasm_worker_pool.ts
//
// WasmWorkerPool — pool di worker WASM con dispatch basato su soglia.
//
// LOGICA DI SELEZIONE (matmul):
//   M * N  <  WASM_MT_THRESHOLD  → single-thread WASM (WasmBridge, main thread)
//   M * N  >= WASM_MT_THRESHOLD  → multi-thread WASM workers
//
//   Esempio concreto:
//     200×200 = 40 000 < 250 000  → WASM single-thread  (matmul SIMD+tiling)
//     600×600 = 360 000 >= 250 000 → WASM worker pool   (matmulChunk parallelo)
//
// LOGICA DI SELEZIONE (jacobi):
//   n²  <  WASM_MT_THRESHOLD  → WASM single-thread (jacobiSolve, zero GC)
//   n²  >= WASM_MT_THRESHOLD  → TS workers paralleli (jacobi SAB zero-copy)
//   Nota: il loop jacobi in WASM è già altamente ottimizzato (SIMD, diagInv
//   precompilato) — non conviene il multi-thread WASM per iterazioni brevi.
//
// INTERFACCIA:
//   Identica a WorkerPool per drop-in replacement in:
//     compute.ts, multiply.ts, jacobi.ts, WorkersBackend.ts
//
// WORKERS:
//   Tipo: wasm_worker.ts (carica WasmBridge + supporta CMD_JACOBI legacy)
//   Numero: floor(cpus/2) con minimo 2
//
// SOGLIA PERSONALIZZABILE:
//   WasmWorkerPool.threshold = 500 * 500  (default)
//   Modificabile a runtime: WasmWorkerPool.threshold = 300 * 300
// ============================================================

import { Worker } from "node:worker_threads";
import * as os    from "node:os";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { WasmBridge, getBridgeSync, initWasm } from "../wasm/wasm_bridge.js";

// ── Costanti ─────────────────────────────────────────────────────────────────
const NUM_WORKERS = Math.max(2, Math.floor(os.cpus().length / 2));

/** Comandi per i worker WASM */
const CMD_WASM_MATMUL_CHUNK = 10;  // matmul chunk via WASM
const CMD_JACOBI            = 2;   // jacobi iterazione via TS (legacy)

// ── Soglia ───────────────────────────────────────────────────────────────────
/**
 * Numero di elementi (M*N) sopra il quale si attiva il multi-thread WASM.
 *
 * Default: 500 * 500 = 250 000
 *   • 200×200 = 40 000  < 250 000 → single-thread WASM
 *   • 600×600 = 360 000 >= 250 000 → multi-thread WASM workers
 */
export const WASM_MT_THRESHOLD = 200 * 200;

// ── Helpers ──────────────────────────────────────────────────────────────────
function toSAB(arr: Float64Array): SharedArrayBuffer {
    const sab = new SharedArrayBuffer(arr.byteLength);
    new Float64Array(sab).set(arr);
    return sab;
}

// ── WasmWorkerPool ────────────────────────────────────────────────────────────
export class WasmWorkerPool {
    private workers: Worker[] = [];

    /** Istanza globale condivisa (singleton applicativo). */
    static instance: WasmWorkerPool;

    /**
     * Soglia M*N sopra cui si usa il multi-thread WASM.
     * Modificabile a runtime prima della chiamata.
     */
    static threshold: number = WASM_MT_THRESHOLD;

    // ── Inizializzazione ──────────────────────────────────────────────────────
    async init(): Promise<void> {
        const baseDir    = dirname(fileURLToPath(import.meta.url));
        const workerPath = resolve(baseDir, "wasm_worker_bootstrap.mjs");

        for (let i = 0; i < NUM_WORKERS; i++) {
            const w = new Worker(workerPath);

            await new Promise<void>((res, rej) => {
                w.once("message", (msg: { ready: boolean; wasmFailed?: boolean }) => {
                    // Il worker segnala ready anche se WASM non è caricato
                    // (in quel caso usa il fallback TS per matmulChunk).
                    if (msg.wasmFailed) {
                        console.warn(`[WasmWorkerPool] worker ${i}: WASM non disponibile, fallback TS attivo`);
                    }
                    res();
                });
                w.once("error", rej);
            });

            this.workers.push(w);
        }
    }

    /** Termina tutti i worker. */
    shutdown(): void {
        for (const w of this.workers) w.terminate();
        this.workers = [];
    }

    // ── matmul ────────────────────────────────────────────────────────────────
    /**
     * Moltiplicazione matriciale con dispatch automatico:
     *   M*N  < threshold → single-thread WASM (matmul SIMD+tiling)
     *   M*N >= threshold → multi-thread WASM (matmulChunk distribuito)
     */
    async matmul(
        A: Float64Array,
        B: Float64Array,
        M: number, K: number, N: number
    ): Promise<Float64Array> {
        if (M * N < WasmWorkerPool.threshold) {
            console.log("single")
            // ── Path single-thread: WASM nel main thread ──────────────────────
            return this._matmulSingleThreadWasm(A, B, M, K, N);
        }
        
            console.log("multi")
        // ── Path multi-thread: WASM workers ───────────────────────────────────
        return this._matmulMultiThreadWasm(A, B, M, K, N);
    }

    // ── jacobi ────────────────────────────────────────────────────────────────
    /**
     * Solver Jacobi con dispatch automatico:
     *   n² < threshold → WASM single-thread (jacobiSolve SIMD, zero GC)
     *   n² >= threshold → TS workers paralleli (SAB zero-copy, efficiente per iter)
     */
    async jacobi(
        A: Float64Array,
        b: Float64Array,
        diagInv: Float64Array,
        n: number,
        tol: number,
        maxIter: number
    ): Promise<Float64Array> {
        if (n * n < WasmWorkerPool.threshold) {
            return this._jacobiSingleThreadWasm(A, b, diagInv, n, tol, maxIter);
        }
        return this._jacobiMultiThreadTS(A, b, diagInv, n, tol, maxIter);
    }

    // ── Implementazioni private ───────────────────────────────────────────────

    /**
     * matmul WASM single-thread nel main thread.
     * Usa WasmBridge.exports.matmul() con SIMD + tiling 64×64.
     */
    private async _matmulSingleThreadWasm(
        A: Float64Array, B: Float64Array,
        M: number, K: number, N: number
    ): Promise<Float64Array> {
        const bridge = await this._getBridge();
        bridge.reset();

        // Alloca prima tutti i buffer (potenziali grow invalidano le view)
        const aPtr = bridge.alloc(M * K);
        const bPtr = bridge.alloc(K * N);
        const cPtr = bridge.alloc(M * N);

        // Copia dati in memoria WASM
        const heap = new Float64Array(bridge.exports.memory.buffer);
        heap.set(A, aPtr >> 3);
        heap.set(B, bPtr >> 3);

        // Esegui matmul WASM (SIMD + tiling 64×64)
        bridge.exports.matmul(aPtr, bPtr, cPtr, M, K, N);

        // Leggi risultato
        const result = new Float64Array(M * N);
        result.set(
            new Float64Array(bridge.exports.memory.buffer, cPtr, M * N)
        );
        bridge.reset();
        return result;
    }

    /**
     * matmul WASM multi-thread: distribuisce le righe di A tra i worker,
     * ognuno esegue matmulChunk() in WASM (SIMD+tiling).
     *
     * Protocollo SAB:
     *   aSAB: A completa (read-only dai worker)
     *   bSAB: B completa (read-only dai worker)
     *   cSAB: output condiviso, ogni worker scrive il suo range [startRow*N .. endRow*N)
     */
    private async _matmulMultiThreadWasm(
        A: Float64Array, B: Float64Array,
        M: number, K: number, N: number
    ): Promise<Float64Array> {
        const aSAB = toSAB(A);
        const bSAB = toSAB(B);
        const cSAB = new SharedArrayBuffer(M * N * 8);

        const chunk = Math.ceil(M / this.workers.length);

        await Promise.all(
            this.workers.map((w, i) => {
                const startRow = i * chunk;
                const endRow   = Math.min(startRow + chunk, M);
                if (startRow >= endRow) return Promise.resolve();

                return new Promise<void>((res) => {
                    const handler = () => { w.off("message", handler); res(); };
                    w.on("message", handler);

                    w.postMessage({
                        cmd: CMD_WASM_MATMUL_CHUNK,
                        aSAB, bSAB, cSAB,
                        M, K, N,
                        startRow, endRow,
                    });
                });
            })
        );

        return new Float64Array(cSAB);
    }

    /**
     * Jacobi WASM single-thread nel main thread.
     * Usa jacobiSolve() che mantiene l'intero loop in WASM:
     *   - SIMD f64x2 per il prodotto riga-vettore
     *   - diagInv precompilato (n mul invece di n div per iterazione)
     *   - zero GC pressure (nessuna allocazione JS per iterazione)
     */
    private async _jacobiSingleThreadWasm(
        A: Float64Array, b: Float64Array, diagInv: Float64Array,
        n: number, tol: number, maxIter: number
    ): Promise<Float64Array> {
        const bridge = await this._getBridge();
        bridge.reset();

        const aPtr    = bridge.alloc(n * n);
        const bPtr    = bridge.alloc(n);
        const xPtr    = bridge.alloc(n);       // soluzione (output)
        const xnPtr   = bridge.alloc(n);       // buffer ping-pong
        const dInvPtr = bridge.alloc(n);       // diagInv (precompilato in WASM)

        const heap = new Float64Array(bridge.exports.memory.buffer);
        heap.set(A, aPtr >> 3);
        heap.set(b, bPtr >> 3);
        // diagInv già calcolato dal chiamante; jacobiSolve lo ricalcola internamente
        // ma passiamo il workspace vuoto (viene sovrascritto)

        // jacobiSolve esegue l'intero loop iterativo in WASM
        bridge.exports.jacobiSolve(
            aPtr, bPtr, xPtr, xnPtr, dInvPtr,
            n, tol, maxIter
        );

        const result = new Float64Array(n);
        result.set(new Float64Array(bridge.exports.memory.buffer, xPtr, n));
        bridge.reset();
        return result;
    }

    /**
     * Jacobi multi-thread con TS workers e SAB zero-copy.
     * Preferito rispetto ai WASM workers per Jacobi perché:
     *   - ogni iterazione richiederebbe copiare A (n×n) in ogni worker WASM
     *   - i TS workers leggono direttamente dai SAB (zero-copy)
     *   - la computazione è ~O(n) per worker per iter → overhead dominante = copia
     */
    private async _jacobiMultiThreadTS(
        A: Float64Array, b: Float64Array, diagInv: Float64Array,
        n: number, tol: number, maxIter: number
    ): Promise<Float64Array> {
        const aSAB    = toSAB(A);
        const bSAB    = toSAB(b);
        const dSAB    = toSAB(diagInv);
        const xSAB    = new SharedArrayBuffer(n * 8);
        const xnSAB   = new SharedArrayBuffer(n * 8);
        const convSAB = new SharedArrayBuffer(this.workers.length * 8);

        const x  = new Float64Array(xSAB);
        const xn = new Float64Array(xnSAB);

        const chunk = Math.ceil(n / this.workers.length);

        for (let iter = 0; iter < maxIter; iter++) {
            await Promise.all(
                this.workers.map((w, i) => {
                    const startRow = i * chunk;
                    const endRow   = Math.min(startRow + chunk, n);
                    if (startRow >= endRow) return Promise.resolve();

                    return new Promise<void>((res) => {
                        const handler = () => { w.off("message", handler); res(); };
                        w.on("message", handler);

                        w.postMessage({
                            cmd: CMD_JACOBI,
                            aSAB, bSAB, xSAB, xnSAB,
                            dSAB, convSAB,
                            n, startRow, endRow, wi: i,
                        });
                    });
                })
            );

            x.set(xn);

            const conv = new Float64Array(convSAB);
            let max = 0;
            for (let i = 0; i < conv.length; i++) {
                if (conv[i] > max) max = conv[i];
            }
            if (max < tol) break;
        }

        return x;
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    /** Ottieni il WasmBridge del main thread (lazy init se necessario). */
    private async _getBridge(): Promise<WasmBridge> {
        let b = getBridgeSync();
        if (!b) {
            await initWasm();
            b = getBridgeSync()!;
        }
        return b;
    }
}

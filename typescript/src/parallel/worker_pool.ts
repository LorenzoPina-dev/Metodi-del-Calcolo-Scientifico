// ============================================================
// src/parallel/worker_pool.ts
//
// WorkerPool singleton — gestisce N worker threads CPU.
// Usa SharedArrayBuffer per comunicazione zero-copy.
//
// Architettura:
//   • Ogni worker vive in un thread separato (worker_threads Node.js)
//   • Controllo via Atomics.wait / Atomics.notify (segnali a 4 byte)
//   • Dati via SharedArrayBuffer (SAB): nessuna serializzazione
//   • Fallback a single-thread se SAB non disponibile (COOP/COEP headers)
//
// Formato SAB di controllo (per pool di N worker):
//   ctrl[0] = stato globale (0=idle, 1=dispatch, 2=shutdown)
//   ctrl[1..N] = stato worker i (0=idle, 1=busy, -1=done)
//   Per ogni worker: slot di 8 i32 = [cmd, startRow, endRow, M, K, N, ...]
//
// ============================================================

import { Worker, workerData, parentPort, isMainThread, receiveMessageOnPort, MessageChannel }
    from "node:worker_threads";
import * as os from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

// ─── Costanti ─────────────────────────────────────────────────────────────────
export const PARALLEL_THRESHOLD = {
    MATMUL   : 90_000,   // n² ≥ 90k  → n ≥ 300 (ammortizza overhead ~0.5ms per worker dispatch)
    JACOBI   : 10_000,   // n² ≥ 10k  → n ≥ 100 (molte iterazioni → overhead fisso irrilevante)
    ELEMENTWISE: 200_000,// elementi  ≥ 200k → n ≥ 450 (mem bandwidth bound)
} as const;

/** Numero di worker = CPU fisiche (non SMT) per evitare contesa cache */
const NUM_WORKERS = Math.max(2, Math.min(8, Math.floor(os.cpus().length / 2)));

// ─── Comandi worker ────────────────────────────────────────────────────────────
export const CMD = {
    IDLE      : 0,
    MATMUL    : 1,  // C[rows] += A[rows] × B   (chunk di righe)
    JACOBI    : 2,  // x_new[rows] = Jacobi step per chunk di righe
    ELEMENTWISE: 3, // op element-wise su chunk di elementi
    SHUTDOWN  : 99,
} as const;

// ─── Layout SharedArrayBuffer di controllo ────────────────────────────────────
// Layout (i32): [globalCmd, worker0State, worker1State, ..., workerN-1State,
//                w0_cmd, w0_p0, w0_p1, w0_p2, w0_p3, w0_p4, w0_p5, w0_p6,
//                w1_cmd, ...]
// Ogni slot worker: 8 × i32 = 32 byte
const CTRL_HEADER = 1 + NUM_WORKERS;  // globalCmd + N state slots
const SLOT_SIZE   = 8;                // i32 per worker slot
const CTRL_SIZE   = CTRL_HEADER + NUM_WORKERS * SLOT_SIZE;

// ─── Struttura del job condiviso ──────────────────────────────────────────────
export interface MatmulJob {
    cmd     : typeof CMD.MATMUL;
    aBuffer : SharedArrayBuffer;   // Float64Array M×K
    bBuffer : SharedArrayBuffer;   // Float64Array K×N
    cBuffer : SharedArrayBuffer;   // Float64Array M×N (output)
    M: number; K: number; N: number;
}

export interface JacobiJob {
    cmd      : typeof CMD.JACOBI;
    aBuffer  : SharedArrayBuffer;   // Float64Array n×n
    bBuffer  : SharedArrayBuffer;   // Float64Array n
    xBuffer  : SharedArrayBuffer;   // Float64Array n (x old, read)
    xnBuffer : SharedArrayBuffer;   // Float64Array n (x new, write)
    diagInvBuffer: SharedArrayBuffer; // Float64Array n
    convBuffer: SharedArrayBuffer;  // Float64Array NUM_WORKERS (maxDiff per worker)
    n: number;
}

// ─── WorkerPool ───────────────────────────────────────────────────────────────
export class WorkerPool {
    private static _instance: WorkerPool | null = null;
    private workers: Worker[] = [];
    private ctrlSAB: SharedArrayBuffer;
    private ctrl: Int32Array;
    private _available = false;
    private _jobData: Map<number, MatmulJob | JacobiJob> = new Map();

    private constructor() {
        this.ctrlSAB = new SharedArrayBuffer(CTRL_SIZE * 4);
        this.ctrl    = new Int32Array(this.ctrlSAB);
    }

    static get instance(): WorkerPool {
        if (!WorkerPool._instance) WorkerPool._instance = new WorkerPool();
        return WorkerPool._instance;
    }

    get available(): boolean { return this._available; }
    get numWorkers(): number { return NUM_WORKERS; }

    async init(): Promise<void> {
        if (this._available) return;
        /*if (!crossOriginIsolated && typeof SharedArrayBuffer !== "undefined") {
            // In Node.js, SAB è sempre disponibile (non serve COOP/COEP)
        }*/
        if (typeof SharedArrayBuffer === "undefined") {
            console.warn("[WorkerPool] SharedArrayBuffer non disponibile. Parallelismo disabilitato.");
            return;
        }

        const workerUrl = resolve(dirname(fileURLToPath(import.meta.url)), "matrix_worker.ts");

        try {
            for (let i = 0; i < NUM_WORKERS; i++) {
                const w = new Worker(workerUrl, {
                    workerData: { workerId: i, ctrlSAB: this.ctrlSAB, numWorkers: NUM_WORKERS },
                    execArgv: process.execArgv.filter(arg => arg !== "--expose-gc")
                });
                w.on("error", (e) => console.error(`[Worker ${i}] errore:`, e));
                this.workers.push(w);
            }
            // Piccola pausa per avvio worker
            await new Promise(r => setTimeout(r, 50));
            this._available = true;
        } catch (e) {
            console.warn("[WorkerPool] Impossibile inizializzare workers:", (e as Error).message);
            this.workers.forEach(w => w.terminate());
            this.workers = [];
        }
    }

    // ─── Matmul parallelo ──────────────────────────────────────────────────────
    // Suddivide le M righe di C tra i worker. Ogni worker calcola C[start..end) × B.
    async matmul(
        aFlat: Float64Array, bFlat: Float64Array, cFlat: Float64Array,
        M: number, K: number, N: number
    ): Promise<void> {
        if (!this._available || this.workers.length === 0) {
            _matmulSerial(aFlat, bFlat, cFlat, M, K, N);
            return;
        }

        // Copia dati in SAB per comunicazione zero-copy
        const aSAB = _toSAB(aFlat);
        const bSAB = _toSAB(bFlat);
        const cSAB = new SharedArrayBuffer(M * N * 8);
        const cView = new Float64Array(cSAB);
        cView.fill(0);

        // Distribuzione righe
        const nw   = Math.min(this.workers.length, M);
        const chunk = Math.ceil(M / nw);
        const promises: Promise<void>[] = [];

        for (let wi = 0; wi < nw; wi++) {
            const startRow = wi * chunk;
            const endRow   = Math.min(startRow + chunk, M);
            if (startRow >= endRow) continue;
            promises.push(this._dispatchMatmulChunk(wi, aSAB, bSAB, cSAB, M, K, N, startRow, endRow));
        }
        await Promise.all(promises);
        cFlat.set(new Float64Array(cSAB));
    }

    // ─── Jacobi parallelo ─────────────────────────────────────────────────────
    // Ogni iterazione: workers calcolano x_new per chunk di righe in parallelo.
    // Convergenza: riduzione del maxDiff tra tutti i worker.
    async jacobi(
        aFlat: Float64Array, bFlat: Float64Array, diagInvFlat: Float64Array,
        n: number, tol: number, maxIter: number
    ): Promise<Float64Array> {
        if (!this._available || this.workers.length === 0) {
            return _jacobiSerial(aFlat, bFlat, diagInvFlat, n, tol, maxIter);
        }

        const aSAB  = _toSAB(aFlat);
        const bSAB  = _toSAB(bFlat);
        const dSAB  = _toSAB(diagInvFlat);
        const xSAB  = new SharedArrayBuffer(n * 8);
        const xnSAB = new SharedArrayBuffer(n * 8);
        const convSAB = new SharedArrayBuffer(NUM_WORKERS * 8);   // maxDiff per worker
        const convMaxAbsSAB = new SharedArrayBuffer(NUM_WORKERS * 8); // maxAbsX per worker

        const xView  = new Float64Array(xSAB);
        const xnView = new Float64Array(xnSAB);
        xView.fill(0); xnView.fill(0);

        const nw   = Math.min(this.workers.length, n);
        const chunk = Math.ceil(n / nw);

        for (let iter = 0; iter < maxIter; iter++) {
            // Reset convergence buffers
            new Float64Array(convSAB).fill(0);
            new Float64Array(convMaxAbsSAB).fill(0);

            // Dispatch tutti i worker in parallelo
            const promises: Promise<void>[] = [];
            for (let wi = 0; wi < nw; wi++) {
                const startRow = wi * chunk;
                const endRow   = Math.min(startRow + chunk, n);
                if (startRow >= endRow) continue;
                promises.push(this._dispatchJacobiChunk(
                    wi, aSAB, bSAB, xSAB, xnSAB, dSAB, convSAB, convMaxAbsSAB,
                    n, startRow, endRow
                ));
            }
            await Promise.all(promises);

            // Swap x ↔ xNew
            xView.set(xnView);

            // Convergenza: max di tutti i worker
            const convView    = new Float64Array(convSAB);
            const convAbsView = new Float64Array(convMaxAbsSAB);
            let maxDiff = 0, maxAbsX = 0;
            for (let wi = 0; wi < nw; wi++) {
                if (convView[wi]    > maxDiff) maxDiff = convView[wi];
                if (convAbsView[wi] > maxAbsX) maxAbsX = convAbsView[wi];
            }
            const denom = maxAbsX > 1.0 ? maxAbsX : 1.0;
            if (maxDiff / denom < tol) break;
        }

        return new Float64Array(xSAB);
    }

    // ─── Dispatch helpers ─────────────────────────────────────────────────────
    private _dispatchMatmulChunk(
        wi: number,
        aSAB: SharedArrayBuffer, bSAB: SharedArrayBuffer, cSAB: SharedArrayBuffer,
        M: number, K: number, N: number, startRow: number, endRow: number
    ): Promise<void> {
        return new Promise((resolve, reject) => {
            const w = this.workers[wi];
            const { port1, port2 } = new MessageChannel();
            port1.once("message", (msg: { done: boolean; error?: string }) => {
                port1.close();
                if (msg.error) reject(new Error(msg.error));
                else resolve();
            });
            w.postMessage(
                { cmd: CMD.MATMUL, aSAB, bSAB, cSAB, M, K, N, startRow, endRow, port: port2 },
                [port2]
            );
        });
    }

    private _dispatchJacobiChunk(
        wi: number,
        aSAB: SharedArrayBuffer, bSAB: SharedArrayBuffer,
        xSAB: SharedArrayBuffer, xnSAB: SharedArrayBuffer,
        dSAB: SharedArrayBuffer, convSAB: SharedArrayBuffer, convMaxAbsSAB: SharedArrayBuffer,
        n: number, startRow: number, endRow: number
    ): Promise<void> {
        return new Promise((resolve, reject) => {
            const w = this.workers[wi];
            const { port1, port2 } = new MessageChannel();
            port1.once("message", (msg: { done: boolean; error?: string }) => {
                port1.close();
                if (msg.error) reject(new Error(msg.error));
                else resolve();
            });
            w.postMessage(
                { cmd: CMD.JACOBI, aSAB, bSAB, xSAB, xnSAB, dSAB, convSAB, convMaxAbsSAB,
                  n, startRow, endRow, wi, port: port2 },
                [port2]
            );
        });
    }

    shutdown(): void {
        this.workers.forEach(w => w.terminate());
        this.workers = [];
        this._available = false;
    }
}

// ─── Helper: Float64Array → SharedArrayBuffer ────────────────────────────────
function _toSAB(arr: Float64Array): SharedArrayBuffer {
    const sab = new SharedArrayBuffer(arr.byteLength);
    new Float64Array(sab).set(arr);
    return sab;
}

// ─── Fallback seriale ─────────────────────────────────────────────────────────
export function _matmulSerial(
    A: Float64Array, B: Float64Array, C: Float64Array,
    M: number, K: number, N: number
): void {
    C.fill(0);
    for (let i = 0; i < M; i++) {
        const iK = i * K, iN = i * N;
        for (let k = 0; k < K; k++) {
            const aik = A[iK + k];
            if (aik === 0) continue;
            const kN = k * N;
            for (let j = 0; j < N; j++) C[iN + j] += aik * B[kN + j];
        }
    }
}

export function _jacobiSerial(
    A: Float64Array, b: Float64Array, diagInv: Float64Array,
    n: number, tol: number, maxIter: number
): Float64Array {
    let x = new Float64Array(n), xNew = new Float64Array(n);
    for (let iter = 0; iter < maxIter; iter++) {
        let maxDiff = 0, maxAbsX = 0;
        for (let i = 0; i < n; i++) {
            const off = i * n; let s = 0;
            for (let j = 0; j < i; j++) s += A[off + j] * x[j];
            for (let j = i + 1; j < n; j++) s += A[off + j] * x[j];
            const xi = (b[i] - s) * diagInv[i];
            xNew[i] = xi;
            const diff = xi - x[i]; const absDiff = diff < 0 ? -diff : diff;
            if (absDiff > maxDiff) maxDiff = absDiff;
            const ax = xi < 0 ? -xi : xi; if (ax > maxAbsX) maxAbsX = ax;
        }
        const tmp = x; x = xNew; xNew = tmp;
        const denom = maxAbsX > 1 ? maxAbsX : 1;
        if (maxDiff / denom < tol) break;
    }
    return x;
}

// ─── Singleton globale ────────────────────────────────────────────────────────
let _pool: WorkerPool | null = null;
let _poolReady = false;

export function getPool(): WorkerPool { return WorkerPool.instance; }

export async function initParallel(): Promise<void> {
    _pool = WorkerPool.instance;
    await _pool.init();
    _poolReady = true;
}

export function getPoolSync(): WorkerPool | null {
    return _poolReady ? WorkerPool.instance : null;
}

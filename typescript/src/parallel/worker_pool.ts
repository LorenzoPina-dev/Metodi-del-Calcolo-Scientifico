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
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const DEBUG = process.env.BENCH_DEBUG === "1";
const DISPATCH_TIMEOUT_MS = Number.parseInt(process.env.BENCH_TIMEOUT_MS ?? "0", 10) || 0;


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

// --- Worker entry + execArgv (tsx) -------------------------------------------
function resolveWorkerEntry(): { path: string; needsTsx: boolean } {
    const baseDir = dirname(fileURLToPath(import.meta.url));
    const tsPath  = resolve(baseDir, "matrix_worker.ts");
    if (existsSync(tsPath)) {
        const bootstrapPath = resolve(baseDir, "matrix_worker_bootstrap.mjs");
        if (existsSync(bootstrapPath)) return { path: bootstrapPath, needsTsx: false };
        return { path: tsPath, needsTsx: true };
    }
    return { path: resolve(baseDir, "matrix_worker.js"), needsTsx: false };
}

function hasTsxHook(args: string[]): boolean {
    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        if (arg === "--loader" || arg === "--import" || arg === "--experimental-loader") {
            const next = args[i + 1] ?? "";
            if (next.includes("tsx")) return true;
        }
        if (arg.startsWith("--loader=") || arg.startsWith("--import=") || arg.startsWith("--experimental-loader=")) {
            if (arg.includes("tsx")) return true;
        }
    }
    return false;
}

function buildWorkerExecArgv(needsTsx: boolean): string[] {
    const base = process.execArgv.filter(arg => arg !== "--expose-gc");
    if (!needsTsx) return base;
    if (hasTsxHook(base)) return base;

    const [maj, min, patch] = process.versions.node.split(".").map(n => Number.parseInt(n, 10));
    const supportsImport =
        (maj > 20) ||
        (maj === 20 && (min > 6 || (min === 6 && patch >= 0))) ||
        (maj === 18 && (min > 19 || (min === 19 && patch >= 0)));
    if (supportsImport) return [...base, "--import", "tsx"];
    return [...base, "--loader", "tsx"];
}

function waitForWorkerReady(w: Worker, index: number, timeoutMs = 5000): Promise<void> {
    return new Promise((resolve, reject) => {
        let done = false;
        const onMessage = (msg: { ready?: boolean }) => {
            if (msg?.ready) {
                cleanup();
                resolve();
            }
        };
        const onError = (err: Error) => {
            cleanup();
            reject(err);
        };
        const onExit = (code: number) => {
            cleanup();
            reject(new Error(`Worker ${index} exited before ready (code ${code})`));
        };
        const timer = setTimeout(() => {
            cleanup();
            reject(new Error(`Worker ${index} ready timeout (${timeoutMs}ms)`));
        }, timeoutMs);

        function cleanup() {
            if (done) return;
            done = true;
            clearTimeout(timer);
            w.off("message", onMessage);
            w.off("error", onError);
            w.off("exit", onExit);
        }

        w.on("message", onMessage);
        w.once("error", onError);
        w.once("exit", onExit);
    });
}

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

        const { path: workerPath, needsTsx } = resolveWorkerEntry();
        const workerExecArgv = buildWorkerExecArgv(needsTsx);

        try {
            const readyPromises: Promise<void>[] = [];
            for (let i = 0; i < NUM_WORKERS; i++) {
                const w = new Worker(workerPath, {
                    workerData: { workerId: i, ctrlSAB: this.ctrlSAB, numWorkers: NUM_WORKERS },
                    execArgv: workerExecArgv,
                    type: "module"
                });
                w.on("error", (e) => console.error(`[Worker ${i}] errore:`, e));
                if (DEBUG) {
                    w.on("message", (msg: { debug?: string; workerId?: number }) => {
                        if (msg?.debug) {
                            const wid = msg.workerId ?? i;
                            console.log(`[Worker ${wid}] ${msg.debug}`);
                        }
                    });
                    w.on("exit", (code) => {
                        console.log(`[Worker ${i}] exit ${code}`);
                    });
                }
                this.workers.push(w);
                readyPromises.push(waitForWorkerReady(w, i));
            }
            await Promise.all(readyPromises);
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
        const settled = await Promise.allSettled(promises);
        const failures = settled.filter((s): s is PromiseRejectedResult => s.status === "rejected");
        if (failures.length) {
            console.error(`[WorkerPool] matmul failed (${failures.length}/${settled.length})`);
            for (const f of failures) {
                const msg = (f.reason && (f.reason as any).message) ? (f.reason as any).message : String(f.reason);
                console.error(`[WorkerPool] matmul error: ${msg}`);
            }
            _matmulSerial(aFlat, bFlat, cFlat, M, K, N);
            return;
        }
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
            let settled = false;
            let timer: NodeJS.Timeout | null = null;

            const finish = (err?: Error) => {
                if (settled) return;
                settled = true;
                if (timer) clearTimeout(timer);
                port1.close();
                if (err) reject(err);
                else resolve();
            };

            port1.once("message", (msg: { done: boolean; error?: string }) => {
                if (msg.error) finish(new Error(`Worker ${wi} matmul error rows=${startRow}-${endRow}: ${msg.error}`));
                else finish();
            });
            port1.start();

            if (DISPATCH_TIMEOUT_MS > 0) {
                timer = setTimeout(() => {
                    finish(new Error(`Worker ${wi} matmul timeout (${DISPATCH_TIMEOUT_MS}ms)`));
                }, DISPATCH_TIMEOUT_MS);
            }

            if (DEBUG) {
                console.log(`[debug] dispatch matmul w=${wi} rows=${startRow}-${endRow}`);
            }

            try {
                w.postMessage(
                    { cmd: CMD.MATMUL, aSAB, bSAB, cSAB, M, K, N, startRow, endRow, port: port2 },
                    [port2]
                );
            } catch (e) {
                finish(e as Error);
            }
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
            let settled = false;
            let timer: NodeJS.Timeout | null = null;

            const finish = (err?: Error) => {
                if (settled) return;
                settled = true;
                if (timer) clearTimeout(timer);
                port1.close();
                if (err) reject(err);
                else resolve();
            };

            port1.once("message", (msg: { done: boolean; error?: string }) => {
                if (msg.error) finish(new Error(`Worker ${wi} jacobi error rows=${startRow}-${endRow}: ${msg.error}`));
                else finish();
            });
            port1.start();

            if (DISPATCH_TIMEOUT_MS > 0) {
                timer = setTimeout(() => {
                    finish(new Error(`Worker ${wi} jacobi timeout (${DISPATCH_TIMEOUT_MS}ms)`));
                }, DISPATCH_TIMEOUT_MS);
            }

            if (DEBUG) {
                console.log(`[debug] dispatch jacobi w=${wi} rows=${startRow}-${endRow}`);
            }

            try {
                w.postMessage(
                    { cmd: CMD.JACOBI, aSAB, bSAB, xSAB, xnSAB, dSAB, convSAB, convMaxAbsSAB,
                      n, startRow, endRow, wi, port: port2 },
                    [port2]
                );
            } catch (e) {
                finish(e as Error);
            }
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

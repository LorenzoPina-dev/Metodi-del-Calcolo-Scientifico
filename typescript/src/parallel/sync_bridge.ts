// ============================================================
// src/parallel/sync_bridge.ts
//
// SyncWorkerBridge — ponte sincrono main thread ↔ worker thread.
//
// OTTIMIZZAZIONI RISPETTO ALLA VERSIONE PRECEDENTE:
//   1. ZERO-COPY SAB: se i Float64Array in ingresso vivono già in un SAB,
//      vengono inviati senza copia (usa lo stesso .buffer).
//   2. POOL INTERNO: il sync_worker usa NUM_WORKERS_INNER sub-worker per
//      matmul → parallelismo reale anche nella via sincrona.
//   3. RIUSO SAB OUTPUT: cSAB/xSAB riutilizzati tra chiamate se le dimensioni
//      non cambiano → 0 allocazioni GC nella fast-path.
//   4. SPINWAIT PRIMA DI Atomics.wait: evita latenza kernel scheduler (≤2ms).
//   5. IMPORT META URL risolto una sola volta al caricamento del modulo.
//
// QUANDO USARE:
//   • matmulSync() / jacobiSync() → path sincrono di mul() / solveJacobiMat()
//     quando WASM non è disponibile o n è in zona grigia.
//   • NON usare in browser (Atomics.wait blocca il main thread UI).
//
// OVERHEAD:
//   • dispatch: ~0.1-0.2ms (SAB + Atomics, zero serializzazione)
//   • Solo per n ≥ PARALLEL_THRESHOLD dove il guadagno supera il costo.
// ============================================================

import { Worker, isMainThread } from "node:worker_threads";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

// ─── Comandi ─────────────────────────────────────────────────────────────────
const CMD_MATMUL   = 1;
const CMD_JACOBI   = 2;
const CMD_SHUTDOWN = 99;

// ─── Path worker (risolto una sola volta) ────────────────────────────────────
const _baseDir = dirname(fileURLToPath(import.meta.url));

function _resolveSyncWorkerPath(): { path: string; needsTsx: boolean } {
    const tsPath = resolve(_baseDir, "sync_worker.ts");
    if (existsSync(tsPath)) {
        const boot = resolve(_baseDir, "sync_worker_bootstrap.mjs");
        if (existsSync(boot)) return { path: boot, needsTsx: false };
        return { path: tsPath, needsTsx: true };
    }
    return { path: resolve(_baseDir, "sync_worker.js"), needsTsx: false };
}

function _buildExecArgv(needsTsx: boolean): string[] {
    const base = process.execArgv.filter(a => a !== "--expose-gc");
    if (!needsTsx) return base;
    for (let i = 0; i < base.length; i++) {
        const a = base[i];
        if ((a === "--loader" || a === "--import") && (base[i + 1] ?? "").includes("tsx")) return base;
        if (a.startsWith("--loader=tsx") || a.startsWith("--import=tsx")) return base;
    }
    const [maj] = process.versions.node.split(".").map(Number);
    return maj >= 20 ? [...base, "--import", "tsx"] : [...base, "--loader", "tsx"];
}

// ─── Stato globale ────────────────────────────────────────────────────────────
let _worker:   Worker | null = null;
let _flagSAB:  SharedArrayBuffer | null = null;
let _flag:     Int32Array | null = null;
let _ready     = false;
let _initProm: Promise<boolean> | null = null;

// SAB di output riutilizzati (zero alloc sulla fast-path)
let _cSAB:    SharedArrayBuffer | null = null;
let _cSABLen  = 0;
let _xSAB:    SharedArrayBuffer | null = null;
let _xSABLen  = 0;

// ─── Inizializzazione ─────────────────────────────────────────────────────────
export function initSyncBridge(): Promise<boolean> {
    if (_ready)    return Promise.resolve(true);
    if (_initProm) return _initProm;

    _initProm = new Promise<boolean>(res => {
        if (!isMainThread)                           { res(false); return; }
        if (typeof SharedArrayBuffer === "undefined") { res(false); return; }

        _flagSAB = new SharedArrayBuffer(4 * 4);
        _flag    = new Int32Array(_flagSAB);

        const { path, needsTsx } = _resolveSyncWorkerPath();
        try {
            _worker = new Worker(path, {
                execArgv: _buildExecArgv(needsTsx),
                type: "module",
            });
            _worker.on("error", e => {
                console.warn("[SyncBridge] worker error:", e.message);
                _ready = false;
            });
            _worker.once("message", (msg: { ready?: boolean }) => {
                _ready = !!msg?.ready;
                res(_ready);
            });
        } catch (e) {
            console.warn("[SyncBridge] impossibile avviare sync worker:", (e as Error).message);
            res(false);
        }
    });

    return _initProm;
}

export function isSyncBridgeReady(): boolean { return _ready; }

// ─── matmulSync ───────────────────────────────────────────────────────────────
export function matmulSync(
    aFlat: Float64Array, bFlat: Float64Array,
    M: number, K: number, N: number
): Float64Array {
    if (!_ready || !_worker || !_flag || !_flagSAB) {
        return _matmulSerialF64(aFlat, bFlat, M, K, N);
    }

    // Zero-copy se già SAB, altrimenti copia minima
    const aSAB = _wrapSAB(aFlat);
    const bSAB = _wrapSAB(bFlat);

    // Riusa cSAB se la dimensione non cambia
    const cLen = M * N;
    if (!_cSAB || _cSABLen !== cLen) {
        _cSAB    = new SharedArrayBuffer(cLen * 8);
        _cSABLen = cLen;
    }

    Atomics.store(_flag, 0, 0);
    Atomics.store(_flag, 2, 0);

    _worker.postMessage({ cmd: CMD_MATMUL, flagSAB: _flagSAB, aSAB, bSAB, cSAB: _cSAB, M, K, N });

    if (!_spinWaitFlag(2000)) {
        console.warn("[SyncBridge] matmulSync timeout — fallback seriale");
        return _matmulSerialF64(aFlat, bFlat, M, K, N);
    }
    if (Atomics.load(_flag, 2) !== 0) return _matmulSerialF64(aFlat, bFlat, M, K, N);

    return new Float64Array(_cSAB);
}

// ─── jacobiSync ───────────────────────────────────────────────────────────────
export function jacobiSync(
    aFlat: Float64Array, bFlat: Float64Array, diagInv: Float64Array,
    n: number, tol: number, maxIter: number
): Float64Array {
    if (!_ready || !_worker || !_flag || !_flagSAB) {
        return _jacobiSerialF64(aFlat, bFlat, diagInv, n, tol, maxIter);
    }

    const aSAB    = _wrapSAB(aFlat);
    const bSAB    = _wrapSAB(bFlat);
    const diagSAB = _wrapSAB(diagInv);

    if (!_xSAB || _xSABLen !== n) {
        _xSAB    = new SharedArrayBuffer(n * 8);
        _xSABLen = n;
    }

    Atomics.store(_flag, 0, 0);
    Atomics.store(_flag, 2, 0);

    _worker.postMessage({
        cmd: CMD_JACOBI, flagSAB: _flagSAB,
        aSAB, bSAB, diagSAB, xSAB: _xSAB,
        n, tol, maxIter,
    });

    if (!_spinWaitFlag(30_000)) {
        console.warn("[SyncBridge] jacobiSync timeout — fallback seriale");
        return _jacobiSerialF64(aFlat, bFlat, diagInv, n, tol, maxIter);
    }
    if (Atomics.load(_flag, 2) !== 0) return _jacobiSerialF64(aFlat, bFlat, diagInv, n, tol, maxIter);

    return new Float64Array(_xSAB);
}

// ─── Shutdown ─────────────────────────────────────────────────────────────────
export function shutdownSyncBridge(): void {
    if (_worker) { _worker.terminate(); _worker = null; }
    _ready     = false;
    _initProm  = null;
    _cSAB      = null;
    _xSAB      = null;
}

// ─── Helpers privati ──────────────────────────────────────────────────────────

/** Spinwait su flag[0] finché ≠ 0, con timeout ms. Ritorna true se ok. */
function _spinWaitFlag(timeoutMs: number): boolean {
    const deadline = Date.now() + Math.min(timeoutMs, 2);  // spinwait max 2ms
    while (Date.now() < deadline) {
        if (Atomics.load(_flag!, 0) !== 0) return true;
    }
    // Poi cede al kernel
    const result = Atomics.wait(_flag!, 0, 0, timeoutMs);
    return result !== "timed-out";
}

/** Zero-copy se già SAB, altrimenti copia in SAB. */
function _wrapSAB(arr: Float64Array): SharedArrayBuffer {
    if (arr.buffer instanceof SharedArrayBuffer) return arr.buffer as SharedArrayBuffer;
    const sab = new SharedArrayBuffer(arr.byteLength);
    new Float64Array(sab).set(arr);
    return sab;
}

function _matmulSerialF64(
    A: Float64Array, B: Float64Array,
    M: number, K: number, N: number
): Float64Array {
    const C = new Float64Array(M * N);
    for (let i = 0; i < M; i++) {
        const iK = i * K, iN = i * N;
        for (let k = 0; k < K; k++) {
            const aik = A[iK + k];
            if (aik === 0) continue;
            const kN = k * N;
            for (let j = 0; j < N; j++) C[iN + j] += aik * B[kN + j];
        }
    }
    return C;
}

function _jacobiSerialF64(
    A: Float64Array, b: Float64Array, diagInv: Float64Array,
    n: number, tol: number, maxIter: number
): Float64Array {
    let x = new Float64Array(n), xNew = new Float64Array(n);
    for (let iter = 0; iter < maxIter; iter++) {
        let maxD = 0, maxA = 0;
        for (let i = 0; i < n; i++) {
            const off = i * n; let s = 0;
            for (let j = 0; j < i; j++)     s += A[off + j] * x[j];
            for (let j = i + 1; j < n; j++) s += A[off + j] * x[j];
            const xi = (b[i] - s) * diagInv[i];
            xNew[i] = xi;
            const d = xi - x[i]; const ad = d < 0 ? -d : d;
            if (ad > maxD) maxD = ad;
            const ax = xi < 0 ? -xi : xi; if (ax > maxA) maxA = ax;
        }
        [x, xNew] = [xNew, x];
        if (maxD / (maxA > 1 ? maxA : 1) < tol) break;
    }
    return x;
}

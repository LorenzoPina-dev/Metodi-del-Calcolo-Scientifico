// ============================================================
// src/parallel/sync_worker.ts
//
// Worker dedicato per operazioni SINCRONE richieste dal main thread.
//
// Protocollo SAB (flag = Int32Array[4]):
//   flag[0] = stato: 0=idle/aspetta, 1=done/ok, 99=shutdown
//   flag[1] = cmd (non usato nel worker, solo lato main)
//   flag[2] = errore: 0=ok, 1=errore
//   flag[3] = padding
//
// OTTIMIZZAZIONI:
//   1. MATMUL CON TRASPOSIZIONE B: il worker riceve B raw e traspone in locale
//      (1 alloc riutilizzabile) → accesso row-row nella dot-product loop.
//   2. LOOP UNROLL ×4: riduce branch overhead nel JIT.
//   3. JACOBI in-place: x e xNew si scambiano con destructuring, 0 alloc/iter.
// ============================================================

import { parentPort } from "node:worker_threads";

const CMD_MATMUL   = 1;
const CMD_JACOBI   = 2;
const CMD_SHUTDOWN = 99;

// ─── Buffer temporaneo per BT (riusato se dimensione non cambia) ──────────────
let _BT: Float64Array | null = null;
let _BTlen = 0;

function _getBT(len: number): Float64Array {
    if (!_BT || _BTlen < len) { _BT = new Float64Array(len); _BTlen = len; }
    return _BT;
}

// ─── Matmul con trasposizione locale di B ─────────────────────────────────────
function _matmul(
    A: Float64Array, B: Float64Array, C: Float64Array,
    M: number, K: number, N: number
): void {
    // Trasponi B → BT
    const BT = _getBT(K * N);
    for (let k = 0; k < K; k++) {
        const kN = k * N;
        for (let j = 0; j < N; j++) BT[j * K + k] = B[kN + j];
    }

    C.fill(0);
    for (let i = 0; i < M; i++) {
        const iK = i * K, iN = i * N;
        for (let j = 0; j < N; j++) {
            const jK = j * K;
            let s = 0.0;
            let k = 0;
            const K4 = K & ~3;
            for (; k < K4; k += 4) {
                s += A[iK + k]     * BT[jK + k]
                   + A[iK + k + 1] * BT[jK + k + 1]
                   + A[iK + k + 2] * BT[jK + k + 2]
                   + A[iK + k + 3] * BT[jK + k + 3];
            }
            for (; k < K; k++) s += A[iK + k] * BT[jK + k];
            C[iN + j] = s;
        }
    }
}

// ─── Jacobi ──────────────────────────────────────────────────────────────────
function _jacobi(
    A: Float64Array, b: Float64Array, diagInv: Float64Array,
    n: number, tol: number, maxIter: number
): Float64Array {
    let x = new Float64Array(n), xNew = new Float64Array(n);
    for (let iter = 0; iter < maxIter; iter++) {
        let maxD = 0, maxA = 0;
        for (let i = 0; i < n; i++) {
            const off = i * n; let s = 0;
            let j = 0;
            const i4 = i & ~3;
            for (; j < i4; j += 4) {
                s += A[off + j]     * x[j]
                   + A[off + j + 1] * x[j + 1]
                   + A[off + j + 2] * x[j + 2]
                   + A[off + j + 3] * x[j + 3];
            }
            for (; j < i; j++) s += A[off + j] * x[j];
            j = i + 1;
            const n4 = n & ~3;
            for (; j < n4; j += 4) {
                s += A[off + j]     * x[j]
                   + A[off + j + 1] * x[j + 1]
                   + A[off + j + 2] * x[j + 2]
                   + A[off + j + 3] * x[j + 3];
            }
            for (; j < n; j++) s += A[off + j] * x[j];

            const xi = (b[i] - s) * diagInv[i];
            xNew[i]  = xi;
            const ad = xi - x[i]; const d = ad < 0 ? -ad : ad;
            if (d > maxD) maxD = d;
            const ax = xi < 0 ? -xi : xi; if (ax > maxA) maxA = ax;
        }
        [x, xNew] = [xNew, x];
        if (maxD / (maxA > 1 ? maxA : 1) < tol) break;
    }
    return x;
}

// ─── Message loop ─────────────────────────────────────────────────────────────
parentPort!.on("message", (msg: {
    cmd:     number;
    flagSAB: SharedArrayBuffer;
    aSAB?:   SharedArrayBuffer;
    bSAB?:   SharedArrayBuffer;
    cSAB?:   SharedArrayBuffer;
    diagSAB?: SharedArrayBuffer;
    xSAB?:   SharedArrayBuffer;
    M?: number; K?: number; N?: number;
    n?: number; tol?: number; maxIter?: number;
}) => {
    const flag = new Int32Array(msg.flagSAB);

    try {
        if (msg.cmd === CMD_MATMUL) {
            const A = new Float64Array(msg.aSAB!);
            const B = new Float64Array(msg.bSAB!);
            const C = new Float64Array(msg.cSAB!);
            _matmul(A, B, C, msg.M!, msg.K!, msg.N!);
            Atomics.store(flag, 2, 0);

        } else if (msg.cmd === CMD_JACOBI) {
            const A       = new Float64Array(msg.aSAB!);
            const b       = new Float64Array(msg.bSAB!);
            const diagInv = new Float64Array(msg.diagSAB!);
            const result  = _jacobi(A, b, diagInv, msg.n!, msg.tol!, msg.maxIter!);
            new Float64Array(msg.xSAB!).set(result);
            Atomics.store(flag, 2, 0);

        } else if (msg.cmd === CMD_SHUTDOWN) {
            Atomics.store(flag, 0, CMD_SHUTDOWN);
            Atomics.notify(flag, 0, 1);
            return;

        } else {
            Atomics.store(flag, 2, 1);  // cmd sconosciuto
        }
    } catch {
        Atomics.store(flag, 2, 1);
    }

    Atomics.store(flag, 0, 1);
    Atomics.notify(flag, 0, 1);
});

parentPort!.postMessage({ ready: true });

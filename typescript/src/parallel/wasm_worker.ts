// ============================================================
// src/parallel/wasm_worker.ts
//
// Worker WASM-powered che sostituisce matrix_worker.ts per matmul.
//
// Comandi supportati:
//   CMD_WASM_MATMUL_CHUNK (10):
//     Riceve la slice A[startRow..endRow) + intera B via SAB,
//     la copia nella memoria WASM, chiama matmulChunk() (SIMD+tiling),
//     e scrive il chunk di output su cSAB[startRow*N..endRow*N).
//
//   CMD_JACOBI (2):
//     Identico a matrix_worker.ts — TS puro, legge direttamente
//     dai SAB senza copia, usato nel path parallelo Jacobi.
//
// Architettura memoria per matmulChunk:
//   - aPtr  = A[startRow*K .. endRow*K)   (slice, non A intera)
//   - bPtr  = B[0 .. K*N)                 (B intera, inevitabile)
//   - cPtr  = chunk output [0..rowCount*N) → scritto su cSAB[startRow*N..]
//   - matmulChunk viene chiamato con startRow=0, endRow=rowCount
//     (indici relativi alla slice) → nessuna dipendenza da M assoluto
//
// Ottimizzazioni:
//   • Slice di A: copia solo le righe necessarie (non l'intera matrice)
//   • reset() prima di ogni matmulChunk: allocator bump O(1)
//   • Jacobi usa SAB zero-copy (Float64Array view diretta)
// ============================================================

import { parentPort, isMainThread } from "node:worker_threads";
import { WasmBridge } from "../wasm/wasm_bridge.js";

if (isMainThread) {
    throw new Error("wasm_worker.ts deve essere eseguito come Worker thread.");
}

// ── Comandi ──────────────────────────────────────────────────────────────────
const CMD_WASM_MATMUL_CHUNK = 10;   // nuovo: chunk matmul via WASM
const CMD_JACOBI            = 2;    // legacy: jacobi via TS (zero-copy SAB)

// ── Stato ────────────────────────────────────────────────────────────────────
let bridge: WasmBridge | null = null;
let busy = false;

// ── Init WASM nel worker ──────────────────────────────────────────────────────
// Ogni worker thread ottiene la propria istanza WASM isolata.
async function init(): Promise<void> {
    try {
        bridge = await WasmBridge.getInstance();
        parentPort!.postMessage({ ready: true });
    } catch (e) {
        console.error("[WasmWorker] init WASM fallito:", e);
        // Segnala comunque ready=true: jacobi TS funziona senza WASM
        parentPort!.postMessage({ ready: true, wasmFailed: true });
    }
}

// ── Loop messaggi ─────────────────────────────────────────────────────────────
parentPort!.on("message", (msg: any) => {
    if (busy) return;
    busy = true;

    try {
        const { cmd } = msg;

        if (cmd === CMD_WASM_MATMUL_CHUNK) {
            _handleMatmulChunk(msg);
        } else if (cmd === CMD_JACOBI) {
            _handleJacobiTS(msg);
        }
    } catch (e) {
        console.error("[WasmWorker] errore cmd", msg?.cmd, ":", e);
    }

    busy = false;
    parentPort!.postMessage({ done: true });
});

// ── CMD_WASM_MATMUL_CHUNK ──────────────────────────────────────────────────────
// Riceve aSAB, bSAB, cSAB + dimensioni + range di righe.
// Esegue matmulChunk() in WASM (SIMD + tiling 64×64).
function _handleMatmulChunk(msg: {
    aSAB: SharedArrayBuffer;
    bSAB: SharedArrayBuffer;
    cSAB: SharedArrayBuffer;
    M: number; K: number; N: number;
    startRow: number; endRow: number;
}): void {
    const { aSAB, bSAB, cSAB, M, K, N, startRow, endRow } = msg;

    // Se WASM non è disponibile → fallback TS
    if (!bridge) {
        _matmulChunkTS(
            new Float64Array(aSAB),
            new Float64Array(bSAB),
            new Float64Array(cSAB),
            M, K, N, startRow, endRow
        );
        return;
    }

    const A = new Float64Array(aSAB);   // view zero-copy
    const B = new Float64Array(bSAB);   // view zero-copy
    const C = new Float64Array(cSAB);   // view zero-copy

    const w = bridge;
    w.reset();  // bump allocator O(1): azzera heapPtr

    const rowCount = endRow - startRow;

    // Alloca WASM memory per: slice A + B intera + output C-chunk
    // Tutti gli alloc PRIMA di accedere alla heap (grow invalida le view).
    const aPtr = w.alloc(rowCount * K);   // slice A[startRow..endRow)
    const bPtr = w.alloc(K * N);          // B intera (necessaria per il prodotto)
    const cPtr = w.alloc(rowCount * N);   // chunk di output

    // Copia dati in WASM memory (refresh heap dopo potenziali grow)
    const heap = new Float64Array(w.exports.memory.buffer);
    heap.set(A.subarray(startRow * K, endRow * K), aPtr >> 3);
    heap.set(B, bPtr >> 3);

    // Zero chunk output prima del calcolo (matmulChunk accumula su C)
    w.exports.zeroF64(cPtr, rowCount * N);

    // matmulChunk(aOff, bOff, cOff, M, K, N, startRow, endRow)
    // aPtr punta alla slice → indici relativi: startRow=0, endRow=rowCount
    w.exports.matmulChunk(aPtr, bPtr, cPtr, M, K, N, 0, rowCount);

    // Copia risultato da WASM memory → cSAB nella posizione corretta
    // (refresh obbligatorio: grow può aver invalidato la view precedente)
    const freshHeap = new Float64Array(w.exports.memory.buffer);
    C.set(
        freshHeap.subarray(cPtr >> 3, (cPtr >> 3) + rowCount * N),
        startRow * N
    );
}

// ── CMD_JACOBI (legacy TS, zero-copy SAB) ────────────────────────────────────
// Identico a matrix_worker.ts — mantiene compatibilità col path parallelo Jacobi.
function _handleJacobiTS(msg: {
    aSAB: SharedArrayBuffer;
    bSAB: SharedArrayBuffer;
    xSAB: SharedArrayBuffer;
    xnSAB: SharedArrayBuffer;
    dSAB: SharedArrayBuffer;
    convSAB: SharedArrayBuffer;
    n: number;
    startRow: number;
    endRow: number;
    wi: number;
}): void {
    const { aSAB, bSAB, xSAB, xnSAB, dSAB, convSAB, n, startRow, endRow, wi } = msg;

    const A       = new Float64Array(aSAB);
    const b       = new Float64Array(bSAB);
    const x       = new Float64Array(xSAB);
    const xn      = new Float64Array(xnSAB);
    const diagInv = new Float64Array(dSAB);
    const conv    = new Float64Array(convSAB);

    let maxDiff = 0;

    for (let i = startRow; i < endRow; i++) {
        const off = i * n;
        let s = 0;

        // j < i (loop separato, zero branch nel ciclo)
        let j = 0;
        for (; j < i; j++) s += A[off + j] * x[j];
        // j > i
        for (j = i + 1; j < n; j++) s += A[off + j] * x[j];

        const xi = (b[i] - s) * diagInv[i];
        xn[i] = xi;

        const d = Math.abs(xi - x[i]);
        if (d > maxDiff) maxDiff = d;
    }

    conv[wi] = maxDiff;
}

// ── Fallback TS per matmulChunk (senza WASM) ──────────────────────────────────
function _matmulChunkTS(
    A: Float64Array, B: Float64Array, C: Float64Array,
    M: number, K: number, N: number,
    startRow: number, endRow: number
): void {
    // Trasponi B per accesso cache-friendly
    const BT = new Float64Array(K * N);
    for (let k = 0; k < K; k++) {
        const kN = k * N;
        for (let j = 0; j < N; j++) BT[j * K + k] = B[kN + j];
    }

    for (let i = startRow; i < endRow; i++) {
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

// ── Avvio ─────────────────────────────────────────────────────────────────────
init();

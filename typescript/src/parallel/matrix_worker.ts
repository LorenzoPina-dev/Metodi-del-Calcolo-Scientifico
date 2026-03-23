// ============================================================
// src/parallel/matrix_worker.ts
//
// Entrypoint del Worker Thread.
// Riceve messaggi dal WorkerPool e calcola chunk di righe.
//
// Operazioni supportate:
//   CMD.MATMUL  — C[startRow..endRow) += A × B (i-k-j cache-friendly)
//   CMD.JACOBI  — x_new[startRow..endRow) = Jacobi step
//
// NOTA: questo file viene eseguito in un worker thread separato.
//       Non ha accesso agli oggetti del thread principale.
//       Comunica solo tramite SharedArrayBuffer e MessagePort.
// ============================================================

import { parentPort, workerData, isMainThread, MessagePort } from "node:worker_threads";

if (isMainThread) {
    // Protegge da import accidentale nel thread principale
    throw new Error("matrix_worker.ts deve essere eseguito come Worker thread.");
}

// ─── Comandi (stesso enum del worker_pool) ────────────────────────────────────
const CMD_MATMUL  = 1;
const CMD_JACOBI  = 2;

// ─── Matmul chunk: calcola C[startRow..endRow) = A[startRow..endRow) × B ─────
// Layout: A è M×K, B è K×N, C è M×N, tutti row-major
function matmulChunk(
    A: Float64Array, B: Float64Array, C: Float64Array,
    M: number, K: number, N: number,
    startRow: number, endRow: number
): void {
    // i-k-j: massimizza riuso cache su B e C
    for (let i = startRow; i < endRow; i++) {
        const iK   = i * K;
        const iN   = i * N;
        for (let k = 0; k < K; k++) {
            const aik = A[iK + k];
            if (aik === 0.0) continue;   // skip esplicito degli zeri (matrici sparse)
            const kN  = k * N;
            // Inner loop j: totalmente sequenziale su B e C → cache L1 friendly
            for (let j = 0; j < N; j++) {
                C[iN + j] += aik * B[kN + j];
            }
        }
    }
}

// ─── Jacobi chunk: calcola x_new[startRow..endRow) ───────────────────────────
// x_new[i] = (b[i] - sum_{j≠i} A[i,j]*x_old[j]) * diagInv[i]
// Usa x_old (immutabile durante l'iterazione → Jacobi parallelo corretto)
function jacobiChunk(
    A       : Float64Array,
    b       : Float64Array,
    x       : Float64Array,   // x old — solo lettura
    xNew    : Float64Array,   // x new — solo scrittura per le righe del chunk
    diagInv : Float64Array,
    n       : number,
    startRow: number,
    endRow  : number,
    wi      : number,
    convDiff   : Float64Array, // output: maxDiff per questo worker
    convMaxAbs : Float64Array  // output: maxAbsX per questo worker
): void {
    let maxDiff = 0.0;
    let maxAbsX = 0.0;

    for (let i = startRow; i < endRow; i++) {
        const off = i * n;
        let s = 0.0;

        // j < i: x old
        for (let j = 0; j < i; j++) s += A[off + j] * x[j];
        // j > i: x old
        for (let j = i + 1; j < n; j++) s += A[off + j] * x[j];

        const xi = (b[i] - s) * diagInv[i];
        xNew[i] = xi;

        const diff   = xi - x[i];
        const absDiff = diff < 0 ? -diff : diff;
        if (absDiff > maxDiff) maxDiff = absDiff;
        const ax = xi < 0 ? -xi : xi;
        if (ax > maxAbsX) maxAbsX = ax;
    }

    // Scrivi risultato convergenza (questo worker ha il suo slot)
    convDiff[wi]   = maxDiff;
    convMaxAbs[wi] = maxAbsX;
}

// ─── Message handler ──────────────────────────────────────────────────────────
parentPort!.on("message", (msg: {
    cmd      : number;
    port     : MessagePort;
    // matmul
    aSAB?    : SharedArrayBuffer;
    bSAB?    : SharedArrayBuffer;
    cSAB?    : SharedArrayBuffer;
    M?: number; K?: number; N?: number;
    startRow?: number;
    endRow?  : number;
    // jacobi
    xSAB?    : SharedArrayBuffer;
    xnSAB?   : SharedArrayBuffer;
    dSAB?    : SharedArrayBuffer;
    convSAB? : SharedArrayBuffer;
    convMaxAbsSAB?: SharedArrayBuffer;
    n?       : number;
    wi?      : number;
}) => {
    const { cmd, port } = msg;

    try {
        if (cmd === CMD_MATMUL) {
            const A = new Float64Array(msg.aSAB!);
            const B = new Float64Array(msg.bSAB!);
            const C = new Float64Array(msg.cSAB!);
            matmulChunk(A, B, C, msg.M!, msg.K!, msg.N!, msg.startRow!, msg.endRow!);
            port.postMessage({ done: true });

        } else if (cmd === CMD_JACOBI) {
            const A    = new Float64Array(msg.aSAB!);
            const b    = new Float64Array(msg.bSAB!);
            const x    = new Float64Array(msg.xSAB!);
            const xNew = new Float64Array(msg.xnSAB!);
            const d    = new Float64Array(msg.dSAB!);
            const conv    = new Float64Array(msg.convSAB!);
            const convAbs = new Float64Array(msg.convMaxAbsSAB!);
            jacobiChunk(A, b, x, xNew, d, msg.n!, msg.startRow!, msg.endRow!, msg.wi!, conv, convAbs);
            port.postMessage({ done: true });

        } else {
            port.postMessage({ done: false, error: `cmd sconosciuto: ${cmd}` });
        }
    } catch (e) {
        port.postMessage({ done: false, error: (e as Error).message });
    } finally {
        port.close();
    }
});

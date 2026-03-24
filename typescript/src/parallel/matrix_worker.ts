import { parentPort, workerData, isMainThread } from "node:worker_threads";

if (isMainThread) {
    throw new Error("matrix_worker.ts deve essere eseguito come Worker thread.");
}

const CMD_MATMUL = 1;
const CMD_JACOBI = 2;

let busy = false;

function matmulChunk(
    A: Float64Array,
    BT: Float64Array, // B trasposta
    C: Float64Array,
    M: number,
    K: number,
    N: number,
    startRow: number,
    endRow: number
) {
    for (let i = startRow; i < endRow; i++) {
        const iK = i * K;
        const iN = i * N;

        for (let j = 0; j < N; j++) {
            const jK = j * K;
            let sum = 0;

            for (let k = 0; k < K; k++) {
                sum += A[iK + k] * BT[jK + k];
            }

            C[iN + j] = sum;
        }
    }
}

function jacobiChunk(
    A: Float64Array,
    b: Float64Array,
    x: Float64Array,
    xNew: Float64Array,
    diagInv: Float64Array,
    n: number,
    startRow: number,
    endRow: number,
    convDiff: Float64Array,
    wi: number
) {
    let maxDiff = 0;

    for (let i = startRow; i < endRow; i++) {
        const off = i * n;
        let s = 0;

        for (let j = 0; j < n; j++) {
            if (j !== i) s += A[off + j] * x[j];
        }

        const xi = (b[i] - s) * diagInv[i];
        xNew[i] = xi;

        const d = Math.abs(xi - x[i]);
        if (d > maxDiff) maxDiff = d;
    }

    convDiff[wi] = maxDiff;
}

parentPort!.on("message", (msg: any) => {
    if (busy) return;
    busy = true;

    const { cmd } = msg;

    if (cmd === CMD_MATMUL) {
        matmulChunk(
            new Float64Array(msg.aSAB),
            new Float64Array(msg.btSAB),
            new Float64Array(msg.cSAB),
            msg.M, msg.K, msg.N,
            msg.startRow, msg.endRow
        );
    } else if (cmd === CMD_JACOBI) {
        jacobiChunk(
            new Float64Array(msg.aSAB),
            new Float64Array(msg.bSAB),
            new Float64Array(msg.xSAB),
            new Float64Array(msg.xnSAB),
            new Float64Array(msg.dSAB),
            msg.n,
            msg.startRow,
            msg.endRow,
            new Float64Array(msg.convSAB),
            msg.wi
        );
    }

    busy = false;
    parentPort!.postMessage({ done: true });
});

parentPort!.postMessage({ ready: true });
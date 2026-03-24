import { Worker } from "node:worker_threads";
import * as os from "node:os";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const NUM_WORKERS = Math.max(2, Math.floor(os.cpus().length / 2));

const CMD = {
    MATMUL: 1,
    JACOBI: 2,
} as const;

function transpose(B: Float64Array, K: number, N: number): Float64Array {
    const BT = new Float64Array(K * N);
    for (let k = 0; k < K; k++) {
        for (let j = 0; j < N; j++) {
            BT[j * K + k] = B[k * N + j];
        }
    }
    return BT;
}

function toSAB(arr: Float64Array): SharedArrayBuffer {
    const sab = new SharedArrayBuffer(arr.byteLength);
    new Float64Array(sab).set(arr);
    return sab;
}

export class WorkerPool {
    private workers: Worker[] = [];
    static instance: WorkerPool;

    async init() {
        const baseDir = dirname(fileURLToPath(import.meta.url));
        // Usa il bootstrap .mjs che registra tsx e poi importa il .ts
        const workerPath = resolve(baseDir, "matrix_worker_bootstrap.mjs");

        for (let i = 0; i < NUM_WORKERS; i++) {
            const w = new Worker(workerPath);

            // Aspetto il primo messaggio dal worker prima di considerarlo pronto
            await new Promise<void>((res, rej) => {
                w.once("message", () => res());
                w.once("error", rej);
            });
            this.workers.push(w);
        }
    }

    shutdown() {
        for (const w of this.workers) {
            w.terminate();
        }
        this.workers = [];
    }

    async matmul(
        A: Float64Array,
        B: Float64Array,
        M: number,
        K: number,
        N: number
    ): Promise<Float64Array> {
        const BT = transpose(B, K, N);

        const aSAB = toSAB(A);
        const btSAB = toSAB(BT);
        const cSAB = new SharedArrayBuffer(M * N * 8);

        const chunk = Math.ceil(M / this.workers.length);

        await Promise.all(
            this.workers.map((w, i) => {
                const start = i * chunk;
                const end = Math.min(start + chunk, M);

                if (start >= end) return Promise.resolve();

                return new Promise<void>((res) => {
                    const handler = () => {
                        w.off("message", handler);
                        res();
                    };

                    w.on("message", handler);

                    w.postMessage({
                        cmd: CMD.MATMUL,
                        aSAB,
                        btSAB,
                        cSAB,
                        M,
                        K,
                        N,
                        startRow: start,
                        endRow: end,
                    });
                });
            })
        );

        return new Float64Array(cSAB);
    }

    async jacobi(
        A: Float64Array,
        b: Float64Array,
        diagInv: Float64Array,
        n: number,
        tol: number,
        maxIter: number
    ): Promise<Float64Array> {
        const aSAB = toSAB(A);
        const bSAB = toSAB(b);
        const dSAB = toSAB(diagInv);

        const xSAB = new SharedArrayBuffer(n * 8);
        const xnSAB = new SharedArrayBuffer(n * 8);
        const convSAB = new SharedArrayBuffer(this.workers.length * 8);

        const x = new Float64Array(xSAB);
        const xn = new Float64Array(xnSAB);

        const chunk = Math.ceil(n / this.workers.length);

        for (let iter = 0; iter < maxIter; iter++) {
            await Promise.all(
                this.workers.map((w, i) => {
                    const start = i * chunk;
                    const end = Math.min(start + chunk, n);

                    if (start >= end) return Promise.resolve();

                    return new Promise<void>((res) => {
                        const handler = () => {
                            w.off("message", handler);
                            res();
                        };

                        w.on("message", handler);

                        w.postMessage({
                            cmd: CMD.JACOBI,
                            aSAB,
                            bSAB,
                            xSAB,
                            xnSAB,
                            dSAB,
                            convSAB,
                            n,
                            startRow: start,
                            endRow: end,
                            wi: i,
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
}

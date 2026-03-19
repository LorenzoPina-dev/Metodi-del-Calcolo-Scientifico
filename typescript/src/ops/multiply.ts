import { Matrix } from "..";

export function multiply(this: Matrix, B: Matrix | number): Matrix {
    // Se B è uno scalare, MATLAB si comporta come un prodotto puntuale
    if (typeof B === "number") {
        const out = new Matrix(this.rows, this.cols);
        const Adat = this.data, Cdat = out.data;
        for (let i = 0; i < Adat.length; i++) Cdat[i] = Adat[i] * B;
        return out;
    }

    if (this.cols !== B.rows) {
        throw new Error(`Inner dimensions must agree: ${this.cols} != ${B.rows}`);
    }

    const M = this.rows, K = this.cols, N = B.cols;
    const out = new Matrix(M, N);
    const Adat = this.data, Bdat = B.data, Cdat = out.data;

    // Ottimizzazione i-k-j per massimizzare l'uso della Cache L1
    for (let i = 0; i < M; i++) {
        const iOff = i * K;
        const outOff = i * N;
        for (let k = 0; k < K; k++) {
            const aik = Adat[iOff + k];
            const kOffB = k * N;
            let j = 0;
            // Loop Unrolling sulle colonne della matrice risultante
            for (; j <= N - 4; j += 4) {
                Cdat[outOff + j] += aik * Bdat[kOffB + j];
                Cdat[outOff + j + 1] += aik * Bdat[kOffB + j + 1];
                Cdat[outOff + j + 2] += aik * Bdat[kOffB + j + 2];
                Cdat[outOff + j + 3] += aik * Bdat[kOffB + j + 3];
            }
            for (; j < N; j++) Cdat[outOff + j] += aik * Bdat[kOffB + j];
        }
    }
    return out;
}
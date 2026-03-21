import { Float64M, Matrix } from "..";

export function multiply(this: Matrix, B: Matrix | number): Matrix {
    // Se B è uno scalare, MATLAB si comporta come un prodotto puntuale
    if (typeof B === "number") {
        const nb = new Float64M(B);
        const out = new Matrix(this.rows, this.cols);
        const Adat = this.data, Cdat = out.data;
        for (let i = 0; i < Adat.length; i++) Cdat[i] = Adat[i].multiply(nb);
        return out;
    }

    if (this.cols !== B.rows) {
        throw new Error(`Inner dimensions must agree: ${this.cols} != ${B.rows}`);
    }

    const { rows:M, cols:K}= this, N = B.cols;
    const out = Matrix.zeros(M, N);
    // Ottimizzazione i-k-j per massimizzare l'uso della Cache L1
    for (let i = 0; i < M; i++) {
        for (let k = 0; k < K; k++) {
            const aik = this.get(i,k);

            let j = 0;
            // Loop Unrolling sulle colonne della matrice risultante
            for (; j <= N - 4; j += 4) {
                out.set(i,j, out.get(i,j).add(aik.multiply(B.get(k, j))));
                out.set(i,j + 1, out.get(i,j + 1).add(aik.multiply(B.get(k, j + 1))));
                out.set(i,j + 2, out.get(i,j + 2).add(aik.multiply(B.get(k, j + 2))));
                out.set(i,j + 3, out.get(i,j + 3).add(aik.multiply(B.get(k, j + 3))));
            }
            for (; j < N; j++) out.set(i,j, out.get(i,j).add(aik.multiply(B.get(k, j))));
        }
    }
    return out;
}
// init/known/hilbert.ts
import { Float64M } from "../..";
import { zeros } from "../init";
import { Matrix } from "../..";

/** Matrice di Hilbert: H(i,j) = 1/(i+j+1). SPD, estremamente mal condizionata. */
export function hilbert(n: number): Matrix<Float64M> {
    const H = zeros(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            H.setNum(i, j, 1 / (i + j + 1));
    return H;
}

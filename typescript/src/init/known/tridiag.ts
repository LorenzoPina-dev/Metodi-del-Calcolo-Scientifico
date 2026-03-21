// init/known/tridiag.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/** Matrice tridiagonale: a = sotto-diagonale, b = diagonale, c = sopra-diagonale. */
export function tridiag(a: number[], b: number[], c: number[]): Matrix<Float64M> {
    const n = b.length;
    const A = zeros(n, n);
    for (let i = 0; i < n; i++) {
        A.setNum(i, i, b[i]);
        if (i > 0)     A.setNum(i, i - 1, a[i - 1]);
        if (i < n - 1) A.setNum(i, i + 1, c[i]);
    }
    return A;
}

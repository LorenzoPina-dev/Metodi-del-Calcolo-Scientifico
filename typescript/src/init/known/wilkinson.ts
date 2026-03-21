// init/known/wilkinson.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/**
 * Matrice di Wilkinson (Wn+): tridiagonale simmetrica con coppie di autovalori quasi identici.
 * Diagonale: |m - i|, m = (n-1)/2. Codiagonali: 1.
 */
export function wilkinson(n: number): Matrix<Float64M> {
    const A = zeros(n, n);
    const m = (n - 1) / 2;
    for (let i = 0; i < n; i++) {
        A.setNum(i, i, Math.abs(m - i));
        if (i < n - 1) {
            A.setNum(i,     i + 1, 1);
            A.setNum(i + 1, i,     1);
        }
    }
    return A;
}

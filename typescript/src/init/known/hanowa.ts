// init/known/hanowa.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/**
 * Matrice di Hanowa (n pari):
 *   [ d*I        -diag(1:m) ]
 *   [ diag(1:m)   d*I       ]
 * Autovalori complessi: d ± i*k, k=1..m.
 */
export function hanowa(n: number, d = -1): Matrix<Float64M> {
    if (n % 2 !== 0) throw new Error("hanowa: n deve essere pari.");
    const m = n / 2;
    const A = zeros(n, n);
    for (let i = 0; i < m; i++) {
        A.setNum(i,     i,     d);
        A.setNum(i + m, i + m, d);
        A.setNum(i,     i + m, -(i + 1));
        A.setNum(i + m, i,      (i + 1));
    }
    return A;
}

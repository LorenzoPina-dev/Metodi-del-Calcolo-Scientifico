// init/known/kahan.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/**
 * Matrice di Kahan: triangolare superiore, dimostra il fallimento del pivoting in QR.
 */
export function kahan(
    n: number,
    m: number = n,
    alpha: number = 1.2,
    pert: number = 1e3
): Matrix<Float64M> {
    const A = zeros(m, n);
    const s = Math.sin(alpha);
    const c = Math.cos(alpha);
    const eps = Number.EPSILON;

    for (let i = 0; i < m; i++) {
        const si  = Math.pow(s, i);
        const csi = -c * si;
        for (let j = 0; j < n; j++) {
            if (j === i)        A.setNum(i, j, si + pert * eps * (Math.min(m, n) - i));
            else if (i < j)     A.setNum(i, j, csi);
        }
    }
    return A;
}

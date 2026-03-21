// init/known/cauchy.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/** Matrice di Cauchy: C(i,j) = 1/(x[i]+y[j]). */
export function cauchy(x: number[], y: number[]): Matrix<Float64M> {
    const n = x.length, m = y.length;
    const C = zeros(n, m);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < m; j++) {
            const d = x[i] + y[j];
            if (Math.abs(d) < 1e-15) throw new Error("cauchy: denominatore zero in (" + i + "," + j + ").");
            C.setNum(i, j, 1 / d);
        }
    return C;
}

// init/known/orthog.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";
import { random } from "../random";

/** Genera una matrice ortogonale casuale n×n tramite Gram-Schmidt. */
export function orthog(n: number): Matrix<Float64M> {
    const A = random(n, n);
    const Q = zeros(n, n);

    for (let j = 0; j < n; j++) {
        const v = Array.from({ length: n }, (_, i) => A.get(i, j).toNumber());

        for (let k = 0; k < j; k++) {
            let dot = 0;
            for (let i = 0; i < n; i++) dot += v[i] * Q.get(i, k).toNumber();
            for (let i = 0; i < n; i++) v[i] -= dot * Q.get(i, k).toNumber();
        }

        const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
        if (norm < 1e-14) throw new Error("orthog: colonna degenere (matrice casuale singolare).");
        for (let i = 0; i < n; i++) Q.setNum(i, j, v[i] / norm);
    }
    return Q;
}

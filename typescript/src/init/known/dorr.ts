// init/known/dorr.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/** Matrice di Dorr: tridiagonale diagonalmente dominante, discretizzazione singolarmente perturbata. */
export function dorr(n: number, theta = 0.01): Matrix<Float64M> {
    const h    = 1 / (n + 1);
    const m    = Math.floor((n + 1) / 2);
    const term = theta / (h * h);

    const c = new Array<number>(n).fill(0);
    const e = new Array<number>(n).fill(0);
    const d = new Array<number>(n).fill(0);

    for (let i = 0; i < m; i++) {
        c[i] = -term;
        e[i] = c[i] - (0.5 - (i + 1) * h) / h;
        d[i] = -(c[i] + e[i]);
    }
    for (let i = m; i < n; i++) {
        e[i] = -term;
        c[i] = e[i] + (0.5 - (i + 1) * h) / h;
        d[i] = -(c[i] + e[i]);
    }

    const A = zeros(n, n);
    for (let i = 0; i < n; i++) {
        A.setNum(i, i, d[i]);
        if (i < n - 1) { A.setNum(i, i + 1, e[i]); A.setNum(i + 1, i, c[i + 1]); }
    }
    return A;
}

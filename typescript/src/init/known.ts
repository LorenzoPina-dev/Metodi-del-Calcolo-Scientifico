import { Matrix } from "../core";
import { zeros } from "./init";

export function hilbert(n: number): Matrix {
    const H = zeros(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            H.set(i, j, 1 / (i + j + 1));
    return H;
}

export function pascal(n: number): Matrix {
    const P = zeros(n, n);
    for (let i = 0; i < n; i++) {
        P.set(i, 0, 1);
        for (let j = 1; j <= i; j++)
            P.set(i, j, P.get(i - 1, j - 1) + P.get(i - 1, j));
    }
    return P;
}

export function magic(n: number): Matrix {
    if (n % 2 === 0) throw new Error("Only odd n implemented");

    const M = zeros(n, n);
    let i = 0, j = Math.floor(n / 2);

    for (let k = 1; k <= n * n; k++) {
        M.set(i, j, k);
        let ni = (i - 1 + n) % n;
        let nj = (j + 1) % n;

        if (M.get(ni, nj) !== 0) i++;
        else { i = ni; j = nj; }
    }

    return M;
}
// init/known/pascal.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/** Matrice di Pascal: A(i,j) = C(i+j, i). SPD, det = 1. */
export function pascal(n: number): Matrix<Float64M> {
    if (n > 30) console.warn("[PRECISION_WARNING] n grande, possibile overflow nei coefficienti binomiali.");
    const A = zeros(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            A.setNum(i, j, nchoosek(i + j, i));
    return A;
}

function nchoosek(n: number, k: number): number {
    if (k < 0 || k > n) return 0;
    if (k === 0 || k === n) return 1;
    if (k > n - k) k = n - k;
    let r = 1;
    for (let i = 0; i < k; i++) { r *= (n - i); r /= (i + 1); }
    return r;
}

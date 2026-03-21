// init/known/binomial.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/** Matrice Binomiale. */
export function binomial(n: number): Matrix<Float64M> {
    const M = zeros(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let k = 0; k < n; k++) {
                const sign = k % 2 === 0 ? 1 : -1;
                sum += sign * combinations(i, k) * combinations(n - 1 - i, j - k);
            }
            M.setNum(i, j, sum);
        }
    return M;
}

function combinations(n: number, k: number): number {
    if (k < 0 || k > n) return 0;
    if (k === 0 || k === n) return 1;
    if (k > n / 2) k = n - k;
    let r = 1;
    for (let i = 1; i <= k; i++) r = r * (n - i + 1) / i;
    return Math.round(r);
}

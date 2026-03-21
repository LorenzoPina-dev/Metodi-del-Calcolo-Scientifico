// init/known/frank.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";
import { triu } from "../../decomposition/triu";
import { minij } from "./minij";

/** Matrice di Frank: Hessenberg superiore con det = 1. */
export function frank(n: number, k = 0): Matrix<Float64M> {
    let F = triu(minij(n), -1);

    if (k === 0) {
        const p = Array.from({ length: n }, (_, i) => n - i - 1);
        const G = zeros(n, n);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++)
                G.set(i, j, F.get(p[i], p[j]));
        // Trasposta
        const H = zeros(n, n);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++)
                H.set(i, j, G.get(j, i));
        F = H;
    }
    return F;
}

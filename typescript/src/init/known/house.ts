// init/known/house.ts
import { Float64M, Matrix } from "../..";
import { identity } from "../init";

/** Matrice di Householder: H = I - 2vv^T/(v^Tv). Simmetrica e ortogonale. */
export function house(x: number[]): Matrix<Float64M> {
    const n = x.length;
    const v = [...x];

    const norm = Math.sqrt(v.reduce((s, xi) => s + xi * xi, 0));
    if (norm === 0) throw new Error("house: vettore nullo.");

    v[0] += Math.sign(v[0] || 1) * norm;   // evita cancellazione
    const beta = v.reduce((s, xi) => s + xi * xi, 0);

    const H = identity(n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
            const cur = H.get(i, j).toNumber();
            H.setNum(i, j, cur - (2 / beta) * v[i] * v[j]);
        }
    return H;
}

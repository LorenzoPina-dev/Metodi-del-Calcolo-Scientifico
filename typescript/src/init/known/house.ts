import { Matrix } from "../..";
import { identity } from "../init";

/**
 * Matrice di Householder
 * * Descrizione:
 * Genera una matrice di riflessione H = I - 2vv^T / (v^T v) che trasforma
 * il vettore x in un multiplo del primo vettore della base canonica e1.
 * * Proprietà:
 * - Simmetrica e Ortogonale (H = H^T = H^-1).
 * - Usata nelle decomposizioni QR e per la riduzione a forma di Hessenberg.
 * * Funzionamento:
 * Calcola il vettore v in modo da evitare la cancellazione numerica (usando il segno di x[1]),
 * quindi applica la trasformazione all'identità.
 */
export function house(x: number[]): Matrix {
    const n = x.length;
    const v = [...x];

    let norm = Math.sqrt(v.reduce((s, xi) => s + xi * xi, 0));
    if (norm === 0) throw new Error("Zero vector");

    v[0] += Math.sign(v[0]) * norm;

    const beta = v.reduce((s, xi) => s + xi * xi, 0);

    const H = identity(n);

    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            H.set(i, j, H.get(i, j) - (2 / beta) * v[i] * v[j]);

    return H;
}
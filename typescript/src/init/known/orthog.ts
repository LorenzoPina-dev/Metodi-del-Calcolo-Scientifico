import { Matrix } from "../../core";
import { zeros } from "../init";
import { random } from "../random";

/**
 * Matrice Ortogonale Casuale
 * * Descrizione:
 * Genera una matrice Q tale che Q^T * Q = I.
 * * Funzionamento:
 * Applica il processo di ortogonalizzazione di Gram-Schmidt modificato a una matrice
 * riempita con valori casuali. Ogni colonna viene proiettata e sottratta dalle successive
 * per garantire l'ortogonalità, poi normalizzata.
 */
export function orthog(n: number): Matrix {
    const A = random(n, n);
    const Q = zeros(n, n);

    for (let j = 0; j < n; j++) {
        let v = Array.from({ length: n }, (_, i) => A.get(i, j));

        for (let k = 0; k < j; k++) {
            let dot = 0;
            for (let i = 0; i < n; i++) dot += v[i] * Q.get(i, k);

            for (let i = 0; i < n; i++) v[i] -= dot * Q.get(i, k);
        }

        let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
        for (let i = 0; i < n; i++) Q.set(i, j, v[i] / norm);
    }

    return Q;
}
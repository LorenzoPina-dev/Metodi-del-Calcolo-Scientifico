import { Matrix } from "../../core";
import { zeros } from "../init";

/**
 * Matrice di Fiedler
 * * Descrizione:
 * Data una sequenza c, la matrice è definita come A(i,j) = |c[i] - c[j]|.
 * * Proprietà:
 * - Simmetrica con diagonale nulla.
 * - Ha esattamente un autovalore positivo, il resto sono negativi.
 * * Funzionamento:
 * Se riceve un numero n, usa il vettore [1, 2, ..., n]. Altrimenti usa il vettore fornito.
 */
export function fiedler(c: number[] | number): Matrix {
    let vec: number[] = typeof c === "number" 
        ? Array.from({ length: c }, (_, i) => i + 1) 
        : c;

    const n = vec.length;
    const A = zeros(n, n);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            A.set(i, j, Math.abs(vec[j] - vec[i]));
        }
    }
    return A;
}
import { Matrix } from "../../core";
import { zeros } from "../init";

/**
 * Matrice di Kahan
 * * Descrizione:
 * Matrice triangolare superiore progettata per illustrare il fallimento del 
 * pivoting nella decomposizione QR per la stima del rango.
 * * Proprietà:
 * - Ha un numero di condizionamento elevato.
 * - Quasi singolare nonostante gli elementi sulla diagonale non siano minuscoli.
 * * Funzionamento:
 * Applica potenze di sin(alpha) sulla diagonale e calsola le sovradiagonali 
 * come -cos(alpha) * sin(alpha)^i.
 */
export function kahan(n: number, m: number = n, alpha: number = 1.2, pert: number = 1e3): Matrix {
    const A = zeros(m, n);
    const s = Math.sin(alpha);
    const c = Math.cos(alpha);
    const eps = Number.EPSILON;

    for (let i = 0; i < m; i++) {
        const si = Math.pow(s, i );
        const csi = -c * si;

        for (let j = 0; j < n; j++) {
            if (j === i) {
                A.set(i, j, si + pert * eps * (Math.min(m, n) - i));
            } else if (i < j) {
                A.set(i, j, csi);
            }
            // sotto diagonale rimane zero
        }
    }

    return A;
}
import { Matrix } from "..";
import { zeros } from "../init";

// Estrae la parte triangolare superiore della matrice con offset k
export function tril(A: Matrix, k = 0): Matrix {
    const n = A.rows;
    const m = A.cols;
    const B = zeros(n, m);
    for (let i = 0; i < A.rows; i++) {
        for (let j = 0; j <= i; j++) {
            B.set(i, j, A.get(i, j));
        }
    }
    return B;
}

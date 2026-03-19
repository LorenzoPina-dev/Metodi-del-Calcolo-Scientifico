import { Matrix } from "..";
import { zeros } from "../init";

// Estrae la parte triangolare superiore della matrice con offset k
export function triu(A: Matrix, k = 0): Matrix {
    const n = A.rows;
    const m = A.cols;
    const B = zeros(n, m);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
            if (j - i >= k) {
                B.set(i, j, A.get(i, j));
            }
        }
    }
    return B;
}
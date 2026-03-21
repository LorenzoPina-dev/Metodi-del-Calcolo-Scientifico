// decomposition/lu.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/**
 * Decomposizione LU senza pivoting.
 * Attenzione: può fallire su pivot nulli; preferire lup() per robustezza.
 */
export function lu<T extends INumeric<T>>(A: Matrix<T>): { L: Matrix<T>; U: Matrix<T> } {
    const n = A.rows;
    if (n !== A.cols) throw new Error("lu: matrice non quadrata.");

    const U = A.clone();
    const L = A.likeIdentity(n);
    const EPS_TOL = 1e-12;

    for (let k = 0; k < n; k++) {
        const pivot = U.get(k, k);
        if (pivot.isNearZero(EPS_TOL)) {
            throw new Error("lu: pivot nullo o quasi-nullo alla colonna " + k + ". Usare lup().");
        }

        for (let i = k + 1; i < n; i++) {
            const factor = U.get(i, k).divide(pivot);
            L.set(i, k, factor);

            for (let j = k; j < n; j++) {
                U.set(i, j, U.get(i, j).subtract(factor.multiply(U.get(k, j))));
            }
        }
    }

    return { L, U };
}

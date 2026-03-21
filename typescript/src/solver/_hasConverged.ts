// solver/_hasConverged.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/**
 * Verifica la convergenza confrontando due iterate successive x e xNext.
 * Usa la norma del massimo (norma infinito) della differenza.
 */
export function _hasConverged<T extends INumeric<T>>(
    oldX: Matrix<T>,
    newX: Matrix<T>,
    tol: number
): boolean {
    for (let i = 0; i < oldX.rows; i++) {
        const diff = newX.get(i, 0).subtract(oldX.get(i, 0)).abs();
        if (!diff.isNearZero(tol)) return false;
    }
    return true;
}

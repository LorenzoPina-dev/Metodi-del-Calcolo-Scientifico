// solver/_hasConverged.ts
import { Matrix } from "..";
import { INumeric } from "../type";

/**
 * Convergenza in norma infinito: max|x_new[i] - x_old[i]| < tol.
 * Float64M fast-path: evita l'allocazione di .subtract().abs().
 */
export function _hasConverged<T extends INumeric<T>>(
    oldX: Matrix<T>, newX: Matrix<T>, tol: number
): boolean {
    const od = oldX.data, nd = newX.data, n = od.length;
    if (oldX.isFloat64) {
        for (let i = 0; i < n; i++) {
            const diff = (nd[i] as any).value - (od[i] as any).value;
            if (diff > tol || diff < -tol) return false;
        }
        return true;
    }
    for (let i = 0; i < n; i++) {
        if (!nd[i].subtract(od[i]).isNearZero(tol)) return false;
    }
    return true;
}

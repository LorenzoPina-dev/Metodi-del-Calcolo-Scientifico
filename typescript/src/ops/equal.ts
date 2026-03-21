// ops/equal.ts
import { Matrix } from "..";
import { INumeric } from "../type";

export function equal<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>, tol = 1e-10): boolean {
    if (A.rows !== B.rows || A.cols !== B.cols) return false;
    const ad = A.data, bd = B.data, len = ad.length;
    // Float64M fast-path: sottrazione inline senza oggetti intermedi
    if (A.isFloat64 && B.isFloat64) {
        for (let i = 0; i < len; i++) {
            const diff = (ad[i] as any).value - (bd[i] as any).value;
            if (diff > tol || diff < -tol) return false;
        }
        return true;
    }
    for (let i = 0; i < len; i++) {
        if (!ad[i].subtract(bd[i]).isNearZero(tol)) return false;
    }
    return true;
}

// ops/equal.ts
import { Matrix } from "..";
import { INumeric } from "../type";

export function equal<T extends INumeric<T>>(A: Matrix<T>, B: Matrix<T>, tol: number = 1e-10): boolean {
    if (A.rows !== B.rows || A.cols !== B.cols) return false;
    for (let i = 0; i < A.data.length; i++) {
        if (!A.data[i].subtract(B.data[i]).isNearZero(tol)) return false;
    }
    return true;
}

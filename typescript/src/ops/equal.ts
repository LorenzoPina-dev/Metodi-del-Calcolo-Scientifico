import { Matrix } from "..";

export function equal(A: Matrix, B: Matrix, tol: number = Matrix.EPS): boolean {
    if (A.rows !== B.rows || A.cols !== B.cols) return false;
    for (let i = 0; i < A.rows; i++) {
        for (let j = 0; j < A.cols; j++) {
            if (Math.abs(A.get(i, j) - B.get(i, j)) > tol) return false;
        }
    }
    return true;
}
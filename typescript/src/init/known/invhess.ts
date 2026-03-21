// init/known/invhess.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/** Matrice Inverse Hessenberg. */
export function invhess(n: number): Matrix<Float64M> {
    const A = zeros(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
            if (j < i)        A.setNum(i, j, j + 1);
            else if (i === j) A.setNum(i, j, i + 1);
            else              A.setNum(i, j, -(i + 1));
        }
    return A;
}

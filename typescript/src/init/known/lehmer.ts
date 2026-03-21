// init/known/lehmer.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/** Matrice di Lehmer: A(i,j) = min(i+1,j+1)/max(i+1,j+1). SPD. */
export function lehmer(n: number): Matrix<Float64M> {
    const A = zeros(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            A.setNum(i, j, Math.min(i + 1, j + 1) / Math.max(i + 1, j + 1));
    return A;
}

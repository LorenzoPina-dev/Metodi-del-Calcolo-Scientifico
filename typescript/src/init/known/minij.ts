// init/known/minij.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/** Matrice Min(i,j): A(i,j) = min(i+1, j+1). SPD. */
export function minij(n: number): Matrix<Float64M> {
    const A = zeros(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            A.setNum(i, j, Math.min(i + 1, j + 1));
    return A;
}

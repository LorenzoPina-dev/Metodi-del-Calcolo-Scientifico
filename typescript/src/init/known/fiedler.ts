// init/known/fiedler.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/**
 * Matrice di Fiedler: A(i,j) = |c[i] - c[j]|.
 * Se si passa un numero n usa il vettore [1,2,...,n].
 * Ha esattamente un autovalore positivo.
 */
export function fiedler(c: number[] | number): Matrix<Float64M> {
    const vec: number[] = typeof c === "number"
        ? Array.from({ length: c }, (_, i) => i + 1)
        : c;
    const n = vec.length;
    const A = zeros(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            A.setNum(i, j, Math.abs(vec[i] - vec[j]));
    return A;
}

// init/known/circul.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/** Matrice circolante: ogni riga è uno shift ciclico della precedente. */
export function circul(v: number[]): Matrix<Float64M> {
    const n = v.length;
    const C = zeros(n, n);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
            C.setNum(i, j, v[(j - i + n) % n]);
    return C;
}

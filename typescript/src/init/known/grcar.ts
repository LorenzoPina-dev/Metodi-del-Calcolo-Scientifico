// init/known/grcar.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/** Matrice di Grcar: Toeplitz non simmetrica con autovalori sensibili alle perturbazioni. */
export function grcar(n: number, k: number = 3): Matrix<Float64M> {
    const G = zeros(n, n);
    for (let i = 0; i < n; i++) {
        G.setNum(i, i, 1);
        for (let j = 1; j <= k; j++)
            if (i + j < n) G.setNum(i, i + j, 1);
        if (i > 0) G.setNum(i, i - 1, -1);
    }
    return G;
}

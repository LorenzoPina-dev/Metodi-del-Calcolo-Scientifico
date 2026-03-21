// init/known/neumann.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/** Matrice di Neumann: Laplaciano discreto con condizioni di Neumann. Singolare. */
export function neumann(n: number): Matrix<Float64M> {
    const k = Math.sqrt(n);
    if (!Number.isInteger(k)) throw new Error("neumann: n deve essere un quadrato perfetto.");
    const A = zeros(n, n);

    for (let p = 1; p <= n; p++) {
        const i = Math.ceil(p / k);
        const j = ((p - 1) % k) + 1;

        // Aggiunge un valore numerico all'elemento (r,c)
        const acc = (r: number, c: number, delta: number) => {
            A.setNum(r, c, A.get(r, c).toNumber() + delta);
        };

        acc(p - 1, p - 1, 4);

        if (k > 1) {
            if (j === 1)      acc(p - 1, p,     -2);
            else if (j === k) acc(p - 1, p - 2, -2);
            else { acc(p - 1, p - 2, -1); acc(p - 1, p, -1); }
        }
        if (k > 1) {
            if (i === 1)      acc(p - 1, p + k - 1, -2);
            else if (i === k) acc(p - 1, p - k - 1, -2);
            else { acc(p - 1, p - k - 1, -1); acc(p - 1, p + k - 1, -1); }
        }
    }
    return A;
}

// init/known/wathen.ts
import { Float64M, Matrix } from "../..";
import { zeros } from "../init";

/** Matrice di Wathen: sparsa SPD da elementi finiti isoparametrici a 8 nodi. */
export function wathen(nx: number, ny: number): Matrix<Float64M> {
    const n = 3 * nx * ny + 2 * nx + 2 * ny + 1;
    const A = zeros(n, n);

    const em = [
        [ 6, -6,  2, -8,  3, -8,  2, -6],
        [-6, 32, -6, 20, -8, 16, -8, 20],
        [ 2, -6,  6, -6,  2, -8,  3, -8],
        [-8, 20, -6, 32, -6, 20, -8, 16],
        [ 3, -8,  2, -6,  6, -6,  2, -8],
        [-8, 16, -8, 20, -6, 32, -6, 20],
        [ 2, -8,  3, -8,  2, -6,  6, -6],
        [-6, 20, -8, 16, -8, 20, -6, 32]
    ];

    const node = new Array<number>(8);

    for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
            node[0] = 3 * (j + 1) * nx + 2 * (j + 1) + 2 * i + 2;
            node[1] = node[0] - 1;
            node[2] = node[0] - 2;
            node[3] = (3 * (j + 1) - 1) * nx + 2 * j + i + 1;
            node[7] = node[3] + 1;
            node[4] = 3 * j * nx + 2 * j + 2 * i;
            node[5] = node[4] + 1;
            node[6] = node[4] + 2;

            const rho = 100 * Math.random();

            for (let kr = 0; kr < 8; kr++)
                for (let kc = 0; kc < 8; kc++) {
                    const r = node[kr], c = node[kc];
                    A.setNum(r, c, A.get(r, c).toNumber() + rho * em[kr][kc]);
                }
        }
    }
    return A;
}

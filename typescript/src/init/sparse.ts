import { Matrix } from "../core";
import { zeros } from "./init";

type SparseOptions =
    | { type: "coo"; rows: number[]; cols: number[]; values: number[] }
    | { type: "random"; density: number; min?: number; max?: number }
    | { type: "diag"; values: number[]; k?: number };

export function sparse(n: number, m: number, opts: SparseOptions): Matrix {
    const S = zeros(n, m);

    if (opts.type === "coo") {
        const { rows, cols, values } = opts;
        for (let i = 0; i < values.length; i++) {
            S.set(rows[i], cols[i], values[i]);
        }
    }

    else if (opts.type === "random") {
        const { density, min = 0, max = 1 } = opts;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < m; j++) {
                if (Math.random() < density) {
                    S.set(i, j, min + Math.random() * (max - min));
                }
            }
        }
    }

    else if (opts.type === "diag") {
        const { values, k = 0 } = opts;
        for (let i = 0; i < values.length; i++) {
            const j = i + k;
            if (j >= 0 && j < m) S.set(i, j, values[i]);
        }
    }

    return S;
}
// init/random.ts
import { Matrix } from "..";
import { Float64M } from "../type";
import { zeros } from "./init";

type RandomOptions = {
    type?: "uniform" | "normal" | "int";
    min?: number;
    max?: number;
    mean?: number;
    std?: number;
};

/** Genera una matrice Float64M con valori casuali. */
export function random(n: number, m: number, opts: RandomOptions = {}): Matrix{
    const { type = "uniform", min = 0, max = 1, mean = 0, std = 1 } = opts;
    const M = zeros(n, m);

    for (let i = 0; i < n * m; i++) {
        let v: number;
        if (type === "uniform") {
            v = min + (max - min) * Math.random();
        } else if (type === "normal") {
            const u1 = Math.random(), u2 = Math.random();
            v = mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        } else { // "int"
            v = Math.floor(min + Math.random() * (max - min + 1));
        }
        M.data[i] = new Float64M(v);
    }
    return M;
}

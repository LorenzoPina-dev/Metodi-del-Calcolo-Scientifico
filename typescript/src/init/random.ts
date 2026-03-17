import { Matrix } from "../core";
import { zeros } from "./init";

type RandomOptions = {
    type?: "uniform" | "normal" | "int";
    min?: number;
    max?: number;
    mean?: number;
    std?: number;
};

export function random(n: number, m: number, opts: RandomOptions = {}): Matrix {
    const { type = "uniform", min = 0, max = 1, mean = 0, std = 1 } = opts;

    const M = zeros(n, m);

    for (let i = 0; i < n * m; i++) {
        if (type === "uniform") {
            M.data[i] = min + (max - min) * Math.random();
        } else if (type === "normal") {
            const u1 = Math.random();
            const u2 = Math.random();
            const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            M.data[i] = mean + std * z;
        } else if (type === "int") {
            M.data[i] = Math.floor(min + Math.random() * (max - min + 1));
        }
    }

    return M;
}
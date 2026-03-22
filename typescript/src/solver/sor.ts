// solver/sor.ts
//
// SOR — Successive Over-Relaxation
//
// Generalizza Gauss-Seidel con il parametro di rilassamento ω ∈ (0, 2):
//   ω = 1  → Gauss-Seidel classico
//   ω < 1  → sotto-rilassamento (rafforza la stabilità)
//   ω > 1  → sovra-rilassamento (accelera la convergenza se ben scelto)
//
// Formula:
//   x_i^{k+1} = (1 - ω) * x_i^k  +  ω * (b_i - Σ_{j<i} a_{ij} x_j^{k+1}
//                                              - Σ_{j>i} a_{ij} x_j^k) / a_{ii}
//
// Converge garantito se A è SPD e 0 < ω < 2.
// Valore ottimale teorico per matrici tridiagonali SPD:
//   ω_opt = 2 / (1 + sqrt(1 - ρ(B_J)²))
// dove ρ(B_J) è il raggio spettrale dell'iterazione di Jacobi.
//
import { Matrix } from "..";
import { INumeric } from "../type";
import { _hasConverged } from "./_hasConverged";

export function solveSOR<T extends INumeric<T>>(
    A: Matrix<T>,
    b: Matrix<T>,
    omega: number = 1.5,
    tol: number = 1e-10,
    maxIter: number = 2000
): Matrix<T> {
    const n = A.rows;
    if (n !== A.cols) throw new Error("solveSOR: matrice non quadrata.");
    if (omega <= 0 || omega >= 2)
        throw new Error(`solveSOR: ω deve essere in (0, 2), ricevuto ${omega}.`);

    const ad = A.data;
    const bd = b.data;
    const omegaT = A.zero.fromNumber(omega);
    const oneMinusOmegaT = A.zero.fromNumber(1 - omega);
    let x = A.like(n, 1);

    for (let iter = 0; iter < maxIter; iter++) {
        const xPrev = x.clone();
        const xd = x.data;

        for (let i = 0; i < n; i++) {
            const off = i * n;
            let s = bd[i];
            for (let j = 0; j < i; j++)      s = s.subtract(ad[off + j].multiply(xd[j]));
            for (let j = i + 1; j < n; j++)   s = s.subtract(ad[off + j].multiply(xd[j]));
            // x_i = (1 - ω)*x_i^old + ω * s / a_{ii}
            const gsStep = s.divide(ad[off + i]);
            xd[i] = oneMinusOmegaT.multiply(xd[i]).add(omegaT.multiply(gsStep));
        }

        if (_hasConverged(xPrev, x, tol)) return x;
    }
    return x;
}

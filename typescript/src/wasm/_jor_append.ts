// ─── 16. JOR (Jacobi Over-Relaxation) ────────────────────────────────────────
// Formula:
//   jacobiStep_i = (b_i - Σ_{j≠i} a_ij * x_j^k) / a_ii
//   x_i^{k+1}   = (1 - omega) * x_i^k + omega * jacobiStep_i
//
// xNewOff è workspace (ping-pong). Ritorna iterazioni usate, -1=non converge.

export function jorSolve(
    aOff: i32, bOff: i32, xOff: i32, xNewOff: i32,
    n: i32, omega: f64, tol: f64, maxIter: i32
): i32 {
    zeroF64(xOff, n);
    const oneMinOmega: f64 = 1.0 - omega;

    for (let iter: i32 = 0; iter < maxIter; iter++) {
        let maxDiff: f64 = 0.0;

        for (let i: i32 = 0; i < n; i++) {
            const rowOff: i32 = aOff + i * n * 8;

            // s = b[i] - Σ_{j < i} a[i,j]*x[j]  (SIMD)
            let s: f64 = ld(bOff, i);
            let j: i32 = 0;
            for (; j + 1 < i; j += 2) {
                const off: i32 = j << 3;
                const v: v128 = f64x2.mul(ld2(rowOff, off), ld2(xOff, off));
                s -= f64x2.extract_lane(v, 0) + f64x2.extract_lane(v, 1);
            }
            for (; j < i; j++) s -= ld(aOff, i * n + j) * ld(xOff, j);

            // s -= Σ_{j > i} a[i,j]*x[j]  (SIMD)
            j = i + 1;
            for (; j + 1 < n; j += 2) {
                const off: i32 = j << 3;
                const v: v128 = f64x2.mul(ld2(rowOff, off), ld2(xOff, off));
                s -= f64x2.extract_lane(v, 0) + f64x2.extract_lane(v, 1);
            }
            for (; j < n; j++) s -= ld(aOff, i * n + j) * ld(xOff, j);

            // JOR update: x_new[i] = (1-omega)*x_old[i] + omega*(s/a[i,i])
            const jacobiStep: f64 = s / ld(aOff, i * n + i);
            const xi: f64 = oneMinOmega * ld(xOff, i) + omega * jacobiStep;
            st(xNewOff, i, xi);

            const diff: f64 = fAbs(xi - ld(xOff, i));
            if (diff > maxDiff) maxDiff = diff;
        }

        // swap: copia xNew → x
        for (let i: i32 = 0; i < n; i++) st(xOff, i, ld(xNewOff, i));

        if (maxDiff < tol) return iter + 1;
    }
    return -1;
}

// solver/cg.ts
//
// CG — Metodo del Gradiente Coniugato (Hestenes & Stiefel, 1952)
//
// Ottimale per A SPD: converge in ≤ n iterazioni in aritmetica esatta.
// In pratica molto meno se gli autovalori sono raggruppati.
//
// BUG CORRETTO rispetto alla versione precedente:
//   p = r + β*p  veniva calcolato DUE volte:
//     1) p = _axpy(r, β, p_old)          → p diventa r + β*p_old  [corretto]
//     2) pd[i] = rd[i] + β * pd[i]       → sovrascrive con r + β*(r + β*p_old) [SBAGLIATO]
//   Fix: un solo aggiornamento inline diretto.
//
// FAST-PATH Float64M:
//   _dotF64   — prodotto interno senza allocazioni
//   _axpyF64  — y + c*x senza allocazioni di oggetti
//   _matvecF64 — A*p (matrix-vector) con Float64Array, O(n²) ottimizzato
//
import { Matrix } from "..";
import { INumeric } from "../type";

export function solveCG<T extends INumeric<T>>(
    A: Matrix<T>,
    b: Matrix<T>,
    tol: number = 1e-10,
    maxIter: number = 2000
): Matrix<T> {
    const n = A.rows;
    if (n !== A.cols) throw new Error("solveCG: matrice non quadrata.");

    if (A.isFloat64) return _cgF64(A as any, b as any, tol, maxIter) as any;
    return _cgGeneric(A, b, tol, maxIter);
}

// ============================================================
// Float64M fast-path — zero allocazioni per iterazione
// ============================================================

function _cgF64(A: Matrix<any>, b: Matrix<any>, tol: number, maxIter: number): Matrix<any> {
    const n = A.rows;
    const ad = A.data;

    // Vettori come Float64Array
    const x  = new Float64Array(n);
    const r  = new Float64Array(n);
    const p  = new Float64Array(n);
    const Ap = new Float64Array(n);

    // r = b, p = b  (x₀ = 0)
    for (let i = 0; i < n; i++) r[i] = p[i] = (b.data[i] as any).value;

    let rho = _dotF64(r, r, n);
    const tol2 = tol * tol;

    for (let iter = 0; iter < maxIter; iter++) {
        if (rho < tol2) break;

        // Ap = A * p  (matrix-vector ottimizzato)
        _matvecF64(ad, p, Ap, n);

        // α = ρ / (p · Ap)
        const pAp = _dotF64(p, Ap, n);
        if (Math.abs(pAp) < 1e-300) break;
        const alpha = rho / pAp;

        // x += α * p,  r -= α * Ap
        for (let i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        const rhoNew = _dotF64(r, r, n);
        const beta   = rhoNew / rho;
        rho = rhoNew;

        // p = r + β * p  (una sola assegnazione)
        for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    }

    // Wrap in Matrix<Float64M>
    const out = A.like(n, 1);
    const F0  = A.zero;
    for (let i = 0; i < n; i++) out.data[i] = F0.fromNumber(x[i]);
    return out;
}

function _dotF64(u: Float64Array, v: Float64Array, n: number): number {
    let s = 0;
    for (let i = 0; i < n; i++) s += u[i] * v[i];
    return s;
}

function _matvecF64(ad: any[], p: Float64Array, out: Float64Array, n: number): void {
    out.fill(0);
    for (let i = 0; i < n; i++) {
        const off = i * n;
        let s = 0;
        for (let j = 0; j < n; j++) s += (ad[off+j] as any).value * p[j];
        out[i] = s;
    }
}

// ============================================================
// Path generico (Complex, Rational)
// ============================================================

function _cgGeneric<T extends INumeric<T>>(
    A: Matrix<T>, b: Matrix<T>, tol: number, maxIter: number
): Matrix<T> {
    const n = A.rows;
    let x  = A.like(n, 1);
    let r  = b.clone() as Matrix<T>;
    let p  = b.clone() as Matrix<T>;

    let rho  = _dotGeneric(r, r);
    const tol2 = tol * tol;

    for (let iter = 0; iter < maxIter; iter++) {
        if (rho < tol2) break;

        const Ap  = A.mul(p);
        const pAp = _dotGeneric(p, Ap);
        if (Math.abs(pAp) < 1e-300) break;

        const alpha  = rho / pAp;
        const alphaT = A.zero.fromNumber(alpha);
        const malphaT = A.zero.fromNumber(-alpha);

        // x += α*p,  r -= α*Ap
        const xd = x.data, rd = r.data, pd = p.data, apd = Ap.data;
        for (let i = 0; i < n; i++) {
            xd[i] = xd[i].add(alphaT.multiply(pd[i]));
            rd[i] = rd[i].add(malphaT.multiply(apd[i]));
        }

        const rhoNew = _dotGeneric(r, r);
        const beta   = rhoNew / rho;
        const betaT  = A.zero.fromNumber(beta);
        rho = rhoNew;

        // p = r + β*p  (una sola passata)
        for (let i = 0; i < n; i++) pd[i] = rd[i].add(betaT.multiply(pd[i]));
    }
    return x;
}

function _dotGeneric<T extends INumeric<T>>(u: Matrix<T>, v: Matrix<T>): number {
    const ud = u.data, vd = v.data, n = ud.length;
    let s = 0;
    for (let i = 0; i < n; i++) s += ud[i].conjugate().multiply(vd[i]).toNumber();
    return s;
}

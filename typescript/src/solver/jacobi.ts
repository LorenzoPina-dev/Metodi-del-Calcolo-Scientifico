// solver/jacobi.ts
//
// Jacobi element-wise con Float64Array fast-path.
//
// PERCHÉ JACOBI È PIÙ LENTO DI GAUSS-SEIDEL:
//
//   1. CONVERGENZA: Jacobi usa x^k per tutti j nell'iterazione k.
//      Gauss-Seidel usa x^{k+1}[j<i] già aggiornati → converge in ~2× meno iter.
//      Entrambi O(n²)/iter, ma GS ha costante ~0.5.
//
//   2. ALLOCAZIONI (path generico): ogni iter crea un nuovo vettore xNext
//      = n nuovi oggetti T + overhead Matrix. Per Float64M con n=100 e 500 iter
//      → 50.000 oggetti per iter, 25M totali → GC pressure enorme.
//
// FAST-PATH Float64M:
//   - Due Float64Array ping-pong: zero allocazioni nell'inner loop.
//   - diagInv pre-calcolato: 1 divisione per riga invece di 1/iter.
//   - j-loop spezzato in [0,i) e (i,n): elimina il branch `if (j !== i)`.
//   - Convergenza inline: max|xNew[i]-x[i]| senza chiamate esterne.
//
// RISULTATO ATTESO:
//   - Float64M: ~5-10× più veloce (elimina le allocazioni)
//   - Ancora ~2× più iterazioni di GS → non raggiunge GS su stessa matrice
//   - Per matrici diagonalmente dominanti converge comunque in O(n²) iter
//
import { Matrix } from "..";
import { Float64M, INumeric } from "../type";
import { _hasConverged } from "./_hasConverged";

export function solveJacobiMat<T extends INumeric<T>>(
    A: Matrix<T>,
    b: Matrix<T>,
    tol: number = 1e-10,
    maxIter: number = 1000
): Matrix<T> {
    const n = A.rows;
    if (n !== A.cols) throw new Error("solveJacobiMat: matrice non quadrata.");

    if (A.isFloat64) return _jacobiF64(A as any, b as any, tol, maxIter);
    return _jacobiGeneric(A, b, tol, maxIter);
}

// ============================================================
// Float64M fast-path — zero allocazioni per iterazione
// ============================================================

function _jacobiF64(
    A: Matrix<Float64M>, b: Matrix<Float64M>,
    tol: number, maxIter: number
): Matrix<any> {
    const n  = A.rows;
    const ad = A.data;
    const bd = b.data;

    // 1. Appiattimento totale su TypedArray crudi PRIMA del loop iterativo
    // Questo elimina ogni singolo accesso a oggetti all'interno del calcolo pesante.
    const a_raw = new Float64Array(n * n);
    for (let i = 1; i <= n * n; i++) {
        a_raw[i - 1] = ad[i - 1].value;
    }

    const b_raw = new Float64Array(n);
    for (let i = 1; i <= n; i++) {
        b_raw[i - 1] = bd[i - 1].value;
    }

    const diagInv = new Float64Array(n);
    for (let i = 1; i <= n; i++) {
        const idx = (i - 1) * n + (i - 1);
        const d = a_raw[idx];
        if (Math.abs(d) < 1e-300) throw new Error(`Jacobi: pivot nullo alla riga ${i}.`);
        diagInv[i - 1] = 1.0 / d;
    }

    let x    = new Float64Array(n);
    let xNew = new Float64Array(n);

    // Loop Iterativo
    for (let iter = 1; iter <= maxIter; iter++) {
        let maxDiff = 0.0;

        for (let i = 1; i <= n; i++) {
            const off = (i - 1) * n;
            let s = b_raw[i - 1];

            // Senza unrolling: V8 trasformerà questo TypedArray loop in codice macchina iper-ottimizzato
            for (let j = 1; j <= i - 1; j++) {
                s -= a_raw[off + j - 1] * x[j - 1];
            }
            for (let j = i + 1; j <= n; j++) {
                s -= a_raw[off + j - 1] * x[j - 1];
            }

            const valNew = s * diagInv[i - 1];
            xNew[i - 1] = valNew;

            // 2. Convergenza inline (Risparmiamo un intero iteramento O(n))
            const diff = valNew - x[i - 1];
            const absDiff = diff < 0 ? -diff : diff;
            if (absDiff > maxDiff) maxDiff = absDiff;
        }

        // Swap ping-pong (zero allocazioni)
        const tmp = x; x = xNew; xNew = tmp;

        if (maxDiff < tol) break;
    }

    // Ricostruzione finale
    const out = A.like(n, 1);
    const F0  = A.zero;
    for (let i = 1; i <= n; i++) {
        out.data[i - 1] = F0.fromNumber(x[i - 1]);
    }
    return out;
}

// ============================================================
// Path generico (Complex, Rational)
// ============================================================

function _jacobiGeneric<T extends INumeric<T>>(
    A: Matrix<T>, b: Matrix<T>,
    tol: number, maxIter: number
): Matrix<T> {
    const n  = A.rows;
    const ad = A.data;
    const bd = b.data;
    let x = A.like(n, 1);

    for (let iter = 0; iter < maxIter; iter++) {
        const xNext = A.like(n, 1);
        const nd    = xNext.data;
        const xd    = x.data;

        for (let i = 0; i < n; i++) {
            const off = i * n;
            let s = bd[i];
            for (let j = 0; j < i; j++)      s = s.subtract(ad[off + j].multiply(xd[j]));
            for (let j = i + 1; j < n; j++) s = s.subtract(ad[off + j].multiply(xd[j]));
            nd[i] = s.divide(ad[off + i]);
        }

        if (_hasConverged(x, xNext, tol)) return xNext;
        x = xNext;
    }
    return x;
}

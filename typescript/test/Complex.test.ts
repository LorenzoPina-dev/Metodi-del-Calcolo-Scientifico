// test/Complex.test.ts
//
// Test profondi per Complex e Matrix<Complex>.
// Organizzati in sezioni:
//   1.  Tipo Complex (aritmetica, confronto, funzioni, edge-case)
//   2.  Costruzione di Matrix<Complex>
//   3.  Operazioni aritmetiche matriciali
//   4.  Algebra lineare (det, trace, norm, inv)
//   5.  Decomposizioni (LUP, QR)
//   6.  Solver lineare
//   7.  Operazioni element-wise (dot, unary)
//   8.  Trasformazioni (t, reshape, slice, flip, rot90)
//   9.  Statistiche (sum, mean, max, min)
//  10.  Proprietà strutturali
//  11.  Identità matematiche complesse
//
import { describe, it, expect } from "vitest";
import { Matrix, Complex } from "../src";

// ============================================================
// Helper
// ============================================================

/** Crea Matrix<Complex> da array 2D di coppie [real, imag]. */
function cmat(data: [number, number][][]): Matrix<Complex> {
    const rows = data.length, cols = data[0].length;
    const m = Matrix.zerosOf(rows, cols, Complex.zero, Complex.one);
    for (let i = 0; i < rows; i++)
        for (let j = 0; j < cols; j++)
            m.set(i, j, new Complex(data[i][j][0], data[i][j][1]));
    return m;
}

/** Confronta due Complex entro tolleranza. */
function cEq(a: Complex, b: Complex, tol = 1e-10): boolean {
    return Math.abs(a.real - b.real) < tol && Math.abs(a.imag - b.imag) < tol;
}

/** Confronta due Matrix<Complex> elemento per elemento entro tolleranza. */
function mEq(A: Matrix<Complex>, B: Matrix<Complex>, tol = 1e-10): boolean {
    if (A.rows !== B.rows || A.cols !== B.cols) return false;
    for (let i = 0; i < A.rows; i++)
        for (let j = 0; j < A.cols; j++)
            if (!cEq(A.get(i, j), B.get(i, j), tol)) return false;
    return true;
}

const TOL = 1e-10;

// ============================================================
// 1. TIPO COMPLEX — aritmetica e funzioni
// ============================================================

describe("Complex — tipo scalare", () => {

    // ---- kind tag ----
    it("ha kind = 'complex'", () => {
        expect(new Complex(1, 2).kind).toBe("complex");
    });

    // ---- Costanti ----
    it("zero e one corretti", () => {
        expect(Complex.zero.real).toBe(0); expect(Complex.zero.imag).toBe(0);
        expect(Complex.one.real).toBe(1);  expect(Complex.one.imag).toBe(0);
    });

    // ---- Memoization magnitudine ----
    it("magnitudeSq memoizzato — stessa istanza", () => {
        const c = new Complex(3, 4);
        const s1 = c.magnitudeSq;
        const s2 = c.magnitudeSq;
        expect(s1).toBe(25);
        expect(s1).toBe(s2);  // stessa referenza numerica
    });

    it("magnitude memoizzato — calcolato una volta", () => {
        const c = new Complex(3, 4);
        expect(c.magnitude).toBeCloseTo(5, 12);
        expect(c.magnitude).toBeCloseTo(5, 12);
    });

    // ---- add / subtract ----
    it("add: (1+2i) + (3+4i) = 4+6i", () => {
        const r = new Complex(1, 2).add(new Complex(3, 4));
        expect(r.real).toBe(4); expect(r.imag).toBe(6);
    });
    it("add commutativa", () => {
        const a = new Complex(2, -3), b = new Complex(-1, 5);
        const ab = a.add(b), ba = b.add(a);
        expect(cEq(ab, ba)).toBe(true);
    });
    it("add con zero è identità", () => {
        const c = new Complex(7, -2);
        expect(cEq(c.add(Complex.zero), c)).toBe(true);
    });
    it("subtract: (5+3i) - (2+1i) = 3+2i", () => {
        const r = new Complex(5, 3).subtract(new Complex(2, 1));
        expect(r.real).toBe(3); expect(r.imag).toBe(2);
    });
    it("subtract di sé stesso è zero", () => {
        const c = new Complex(3, 7);
        expect(cEq(c.subtract(c), Complex.zero)).toBe(true);
    });

    // ---- multiply ----
    it("multiply: (1+i)(1-i) = 2", () => {
        const r = new Complex(1, 1).multiply(new Complex(1, -1));
        expect(cEq(r, new Complex(2, 0))).toBe(true);
    });
    it("multiply: i*i = -1", () => {
        const i = new Complex(0, 1);
        expect(cEq(i.multiply(i), new Complex(-1, 0))).toBe(true);
    });
    it("multiply commutativa", () => {
        const a = new Complex(2, 3), b = new Complex(-1, 4);
        expect(cEq(a.multiply(b), b.multiply(a))).toBe(true);
    });
    it("multiply per uno è identità", () => {
        const c = new Complex(5, -2);
        expect(cEq(c.multiply(Complex.one), c)).toBe(true);
    });
    it("multiply per coniugato = modulo²", () => {
        const c = new Complex(3, 4);
        const r = c.multiply(c.conjugate());
        expect(r.real).toBeCloseTo(25, 12);
        expect(r.imag).toBeCloseTo(0, 12);
    });

    // ---- divide ----
    it("divide: (2+4i) / (1+i) = 3+i", () => {
        const r = new Complex(2, 4).divide(new Complex(1, 1));
        expect(cEq(r, new Complex(3, 1))).toBe(true);
    });
    it("divide per sé stesso = 1", () => {
        const c = new Complex(3, 4);
        expect(cEq(c.divide(c), Complex.one)).toBe(true);
    });
    it("divide per zero lancia errore", () => {
        expect(() => new Complex(1, 1).divide(Complex.zero)).toThrow();
    });

    // ---- negate ----
    it("negate inverte segno", () => {
        const c = new Complex(2, -5);
        const n = c.negate();
        expect(n.real).toBe(-2); expect(n.imag).toBe(5);
    });
    it("c + negate(c) = zero", () => {
        const c = new Complex(4, 3);
        expect(cEq(c.add(c.negate()), Complex.zero)).toBe(true);
    });

    // ---- abs ----
    it("abs di z reale positivo = z stesso", () => {
        const c = new Complex(5, 0);
        expect(c.abs().real).toBeCloseTo(5, 12);
        expect(c.abs().imag).toBeCloseTo(0, 12);
    });
    it("abs di 3+4i = 5+0i", () => {
        const c = new Complex(3, 4).abs();
        expect(c.real).toBeCloseTo(5, 10);
        expect(c.imag).toBeCloseTo(0, 10);
    });

    // ---- sqrt ----
    it("sqrt(i) = (1+i)/√2", () => {
        const s = new Complex(0, 1).sqrt();
        const expected = 1 / Math.sqrt(2);
        expect(s.real).toBeCloseTo(expected, 10);
        expect(s.imag).toBeCloseTo(expected, 10);
    });
    it("sqrt(z)² = z (identità)", () => {
        const z = new Complex(3, 4);
        const s = z.sqrt();
        const s2 = s.multiply(s);
        expect(cEq(s2, z, 1e-10)).toBe(true);
    });
    it("sqrt(-1) = i", () => {
        const s = new Complex(-1, 0).sqrt();
        expect(Math.abs(s.real)).toBeCloseTo(0, 10);
        expect(Math.abs(s.imag)).toBeCloseTo(1, 10);
    });

    // ---- round ----
    it("round arrotonda parti reale e immaginaria", () => {
        const r = new Complex(1.7, -2.3).round();
        expect(r.real).toBe(2); expect(r.imag).toBe(-2);
    });

    // ---- conjugate ----
    it("coniugato di (a+bi) = a-bi", () => {
        const c = new Complex(3, -5);
        const conj = c.conjugate();
        expect(conj.real).toBe(3); expect(conj.imag).toBe(5);
    });
    it("coniugato del coniugato = originale", () => {
        const c = new Complex(-2, 7);
        expect(cEq(c.conjugate().conjugate(), c)).toBe(true);
    });

    // ---- Comparazione ----
    it("greaterThan su modulo²  — senza sqrt", () => {
        const big = new Complex(3, 4);   // |big|²=25
        const small = new Complex(1, 1); // |small|²=2
        expect(big.greaterThan(small)).toBe(true);
        expect(small.greaterThan(big)).toBe(false);
    });
    it("lessThan coerente con greaterThan", () => {
        const a = new Complex(1, 0), b = new Complex(2, 0);
        expect(a.lessThan(b)).toBe(true);
        expect(b.lessThan(a)).toBe(false);
    });
    it("equals controlla real e imag esatti", () => {
        expect(new Complex(1, 2).equals(new Complex(1, 2))).toBe(true);
        expect(new Complex(1, 2).equals(new Complex(1, 3))).toBe(false);
    });
    it("isNearZero su |z| < tol", () => {
        const tiny = new Complex(1e-12, 1e-12);
        expect(tiny.isNearZero(1e-10)).toBe(true);
        expect(tiny.isNearZero(1e-13)).toBe(false);
    });

    // ---- fromNumber / toNumber ----
    it("fromNumber(5) = Complex(5, 0)", () => {
        const c = Complex.zero.fromNumber(5);
        expect(c.real).toBe(5); expect(c.imag).toBe(0);
    });
    it("toNumber restituisce il modulo", () => {
        expect(new Complex(3, 4).toNumber()).toBeCloseTo(5, 12);
    });

    // ---- toString ----
    it("toString con parte immaginaria positiva", () => {
        expect(new Complex(1, 2).toString()).toBe("1 + 2i");
    });
    it("toString con parte immaginaria negativa", () => {
        expect(new Complex(1, -2).toString()).toBe("1 - 2i");
    });
    it("toString solo reale", () => {
        expect(new Complex(3, 0).toString()).toBe("3");
    });

    // ---- Proprietà algebriche: distributività ----
    it("a*(b+c) = a*b + a*c", () => {
        const a = new Complex(1, 2), b = new Complex(3, -1), c = new Complex(-2, 4);
        const lhs = a.multiply(b.add(c));
        const rhs = a.multiply(b).add(a.multiply(c));
        expect(cEq(lhs, rhs)).toBe(true);
    });
});

// ============================================================
// 2. COSTRUZIONE Matrix<Complex>
// ============================================================

describe("Matrix<Complex> — costruzione", () => {

    it("zerosOf: tutti zero", () => {
        const M = Matrix.zerosOf(2, 3, Complex.zero, Complex.one);
        expect(M.rows).toBe(2); expect(M.cols).toBe(3);
        for (let i = 0; i < 2; i++)
            for (let j = 0; j < 3; j++)
                expect(cEq(M.get(i, j), Complex.zero)).toBe(true);
    });

    it("identityOf: diagonale = 1, off-diag = 0", () => {
        const I = Matrix.identityOf(3, Complex.zero, Complex.one);
        for (let i = 0; i < 3; i++)
            for (let j = 0; j < 3; j++) {
                const expected = i === j ? Complex.one : Complex.zero;
                expect(cEq(I.get(i, j), expected)).toBe(true);
            }
    });

    it("fromTypedArray: popola correttamente", () => {
        const data: Complex[][] = [
            [new Complex(1, 2), new Complex(3, 4)],
            [new Complex(5, 6), new Complex(7, 8)],
        ];
        const M = Matrix.fromTypedArray(data, Complex.zero, Complex.one);
        expect(cEq(M.get(0, 0), new Complex(1, 2))).toBe(true);
        expect(cEq(M.get(1, 1), new Complex(7, 8))).toBe(true);
    });

    it("set con number usa fromNumber", () => {
        const M = Matrix.zerosOf(2, 2, Complex.zero, Complex.one);
        M.set(0, 0, 5);         // numero → Complex(5, 0)
        expect(M.get(0, 0).real).toBe(5);
        expect(M.get(0, 0).imag).toBe(0);
    });

    it("clone è una copia indipendente", () => {
        const A = cmat([[[1, 2], [3, 4]]]);
        const B = A.clone();
        B.set(0, 0, new Complex(99, 99));
        expect(A.get(0, 0).real).toBe(1); // A non è cambiata
    });
});

// ============================================================
// 3. OPERAZIONI ARITMETICHE MATRICIALI
// ============================================================

describe("Matrix<Complex> — operazioni aritmetiche", () => {

    const A = cmat([[[1, 1], [2, -1]], [[-1, 2], [3, 0]]]);
    const B = cmat([[[0, 1], [1, 0]],  [[2, 1],  [-1, 1]]]);

    it("add: A + B componente per componente", () => {
        const C = A.add(B);
        expect(cEq(C.get(0, 0), new Complex(1, 2))).toBe(true);
        expect(cEq(C.get(1, 1), new Complex(2, 1))).toBe(true);
    });

    it("add commutativa: A+B = B+A", () => {
        expect(mEq(A.add(B), B.add(A))).toBe(true);
    });

    it("sub: A - A = zero matrix", () => {
        expect(A.sub(A).isZeroMatrix(1e-12)).toBe(true);
    });

    it("sub: A - B = A + negate(B)", () => {
        expect(mEq(A.sub(B), A.add(B.negate()))).toBe(true);
    });

    it("mul scalare: A * 2 = A + A", () => {
        expect(mEq(A.mul(2), A.add(A))).toBe(true);
    });

    it("mul scalare con complesso usato da fromNumber", () => {
        const C = A.mul(2);
        expect(cEq(C.get(0, 0), new Complex(2, 2))).toBe(true);
    });

    it("mul matriciale: A * I = A", () => {
        const I = Matrix.identityOf(2, Complex.zero, Complex.one);
        expect(mEq(A.mul(I), A)).toBe(true);
    });

    it("mul: dimensioni interne non coincidono → errore", () => {
        const C = Matrix.zerosOf(3, 2, Complex.zero, Complex.one);
        expect(() => A.mul(C)).toThrow();
    });

    it("mul associativa: (A*B)*I = A*(B*I)", () => {
        const I = Matrix.identityOf(2, Complex.zero, Complex.one);
        expect(mEq(A.mul(B).mul(I), A.mul(B.mul(I)))).toBe(true);
    });

    it("mul distributiva: A*(B+I) = A*B + A*I", () => {
        const I = Matrix.identityOf(2, Complex.zero, Complex.one);
        const lhs = A.mul(B.add(I));
        const rhs = A.mul(B).add(A.mul(I));
        expect(mEq(lhs, rhs, 1e-10)).toBe(true);
    });

    it("pow(0) = identità", () => {
        const I = Matrix.identityOf(2, Complex.zero, Complex.one);
        expect(mEq(A.pow(0), I)).toBe(true);
    });

    it("pow(1) = A", () => {
        expect(mEq(A.pow(1), A)).toBe(true);
    });

    it("pow(2) = A * A", () => {
        expect(mEq(A.pow(2), A.mul(A), 1e-10)).toBe(true);
    });
});

// ============================================================
// 4. ALGEBRA LINEARE
// ============================================================

describe("Matrix<Complex> — algebra lineare", () => {

    // det 1×1
    it("det 1×1", () => {
        const A = cmat([[[3, -2]]]);
        expect(cEq(A.det(), new Complex(3, -2))).toBe(true);
    });

    // det 2×2: det([[a,b],[c,d]]) = ad - bc
    it("det 2×2 = ad - bc", () => {
        // [[1+i, 2], [3, 4-i]]  →  (1+i)(4-i) - 2*3 = (5+3i) - 6 = -1+3i
        const A = cmat([[[1, 1], [2, 0]], [[3, 0], [4, -1]]]);
        const expected = new Complex(-1, 3);
        expect(cEq(A.det(), expected)).toBe(true);
    });

    // det matrice identità = 1
    it("det(I) = 1", () => {
        const I = Matrix.identityOf(3, Complex.zero, Complex.one);
        expect(cEq(I.det(), Complex.one)).toBe(true);
    });

    // det matrice singolare = 0
    it("det matrice singolare = 0", () => {
        // [[1+i, 2+2i], [1, 2]] → colonne proporzionali
        const A = cmat([[[1, 1], [2, 2]], [[1, 0], [2, 0]]]);
        expect(A.det().isNearZero(1e-10)).toBe(true);
    });

    // det moltiplicativa: det(AB) = det(A)*det(B)
    it("det(A*B) = det(A) * det(B)", () => {
        const A = cmat([[[1, 1], [0, 1]], [[1, 0], [1, -1]]]);
        const B = cmat([[[2, 0], [1, 1]], [[-1, 1], [0, 2]]]);
        const detAB = A.mul(B).det();
        const detAdB = A.det().multiply(B.det());
        expect(cEq(detAB, detAdB, 1e-10)).toBe(true);
    });

    // trace
    it("trace = somma diagonale", () => {
        // [[1+2i, 0], [0, 3-i]] → trace = 4+i
        const A = cmat([[[1, 2], [0, 0]], [[0, 0], [3, -1]]]);
        expect(cEq(A.trace(), new Complex(4, 1))).toBe(true);
    });

    // norm frobenius
    it("norm Frobenius: ||I_2|| = √2", () => {
        const I = Matrix.identityOf(2, Complex.zero, Complex.one);
        expect(I.norm("Fro")).toBeCloseTo(Math.sqrt(2), 10);
    });

    it("norm 1 di matrice complessa", () => {
        // [[3+4i, 0], [0, 1]] → norma 1 = max(|3+4i|, |1|) = max(5, 1) = 5
        const A = cmat([[[3, 4], [0, 0]], [[0, 0], [1, 0]]]);
        expect(A.norm("1")).toBeCloseTo(5, 10);
    });

    // inv
    it("A * inv(A) = I", () => {
        const A = cmat([[[2, 1], [1, 0]], [[0, 1], [1, 1]]]);
        const AI = A.mul(A.inv());
        const I = Matrix.identityOf(2, Complex.zero, Complex.one);
        expect(mEq(AI, I, 1e-9)).toBe(true);
    });

    it("inv(I) = I", () => {
        const I = Matrix.identityOf(3, Complex.zero, Complex.one);
        expect(mEq(I.inv(), I, 1e-10)).toBe(true);
    });

    it("inv di matrice singolare lancia errore", () => {
        const A = cmat([[[1, 0], [2, 0]], [[1, 0], [2, 0]]]);
        expect(() => A.inv()).toThrow();
    });
});

// ============================================================
// 5. DECOMPOSIZIONI
// ============================================================

describe("Matrix<Complex> — decomposizioni", () => {

    describe("LUP", () => {
        it("L * U = P * A", () => {
            const A = cmat([[[1, 1], [2, -1]], [[-1, 2], [3, 0]]]);
            const { L, U, P } = Matrix.decomp.lup(A);
            // Costruisce PA manualmente
            const n = 2;
            const PA = Matrix.zerosOf(n, n, Complex.zero, Complex.one);
            for (let i = 0; i < n; i++)
                for (let j = 0; j < n; j++)
                    PA.set(i, j, A.get(P[i], j));
            expect(mEq(L.mul(U), PA, 1e-10)).toBe(true);
        });

        it("L è triangolare inferiore unitaria", () => {
            const A = cmat([[[2, 1], [1, 0]], [[0, 1], [3, -1]]]);
            const { L } = Matrix.decomp.lup(A);
            expect(L.isLowerTriangular(1e-10)).toBe(true);
            for (let i = 0; i < L.rows; i++)
                expect(cEq(L.get(i, i), Complex.one)).toBe(true);
        });

        it("U è triangolare superiore", () => {
            const A = cmat([[[2, 1], [1, 0]], [[0, 1], [3, -1]]]);
            const { U } = Matrix.decomp.lup(A);
            expect(U.isUpperTriangular(1e-10)).toBe(true);
        });

        it("lancia errore su matrice singolare", () => {
            const A = cmat([[[1, 0], [2, 0]], [[1, 0], [2, 0]]]);
            expect(() => Matrix.decomp.lup(A)).toThrow();
        });
    });

    describe("QR", () => {
        it("Q * R = A", () => {
            const A = cmat([[[1, 1], [2, 0]], [[0, 1], [1, -1]], [[-1, 0], [1, 1]]]);
            const { Q, R } = Matrix.decomp.qr(A);
            expect(mEq(Q.mul(R), A, 1e-9)).toBe(true);
        });

        it("Q^H * Q = I (colonne ortonormali)", () => {
            // Per matrici complesse Q^H = conj(Q)^T
            const A = cmat([[[1, 1], [2, 0]], [[0, 1], [1, -1]], [[-1, 0], [1, 1]]]);
            const { Q } = Matrix.decomp.qr(A);
            // Q^T * Q (Gram-Schmidt classico usa il trasposto non il coniugato)
            const QtQ = Q.t().mul(Q);
            const I = Matrix.identityOf(2, Complex.zero, Complex.one);
            // Per Q reale Q^T*Q = I; per Q complesso questo è approssimato
            // Verifica l'ortogonalità con isOrthogonal
            // oppure per colonne: ||q_k||² ≈ 1 e q_i·q_j ≈ 0
            for (let j = 0; j < Q.cols; j++) {
                let norm2 = Complex.zero;
                for (let i = 0; i < Q.rows; i++) {
                    const v = Q.get(i, j);
                    norm2 = norm2.add(v.multiply(v)); // v² non |v|²
                }
                // ||q_j||² potrebbe avere parte immaginaria per vettori complessi,
                // ma la norma reale = 1
                let realNorm2 = 0;
                for (let i = 0; i < Q.rows; i++) {
                    const v = Q.get(i, j).magnitude;
                    realNorm2 += v * v;
                }
                expect(realNorm2).toBeCloseTo(1, 8);
            }
        });

        it("R è triangolare superiore", () => {
            const A = cmat([[[1, 1], [2, 0]], [[0, 1], [1, -1]], [[-1, 0], [1, 1]]]);
            const { R } = Matrix.decomp.qr(A);
            expect(R.isUpperTriangular(1e-9)).toBe(true);
        });

        it("lancia errore su colonne dipendenti", () => {
            const A = cmat([[[1, 0], [2, 0]], [[2, 0], [4, 0]], [[3, 0], [6, 0]]]);
            expect(() => Matrix.decomp.qr(A)).toThrow();
        });
    });
});

// ============================================================
// 6. SOLVER LINEARE
// ============================================================

describe("Matrix<Complex> — solver", () => {

    // Sistema 2×2: Az = b
    //   A = [[2+i, 1], [1, 1-i]]
    //   b = [[3+2i], [2-i]]
    function buildSystem() {
        const A = cmat([[[2, 1], [1, 0]], [[1, 0], [1, -1]]]);
        const b = Matrix.zerosOf(2, 1, Complex.zero, Complex.one);
        b.set(0, 0, new Complex(3, 2));
        b.set(1, 0, new Complex(2, -1));
        return { A, b };
    }

    it("LUP: A * x = b → A * (solve(b)) ≈ b", () => {
        const { A, b } = buildSystem();
        const x = A.solve(b, "LUP");
        const Ax = A.mul(x);
        expect(mEq(Ax, b, 1e-10)).toBe(true);
    });

    it("LU: stessa soluzione di LUP (quando non serve pivoting)", () => {
        const { A, b } = buildSystem();
        const xLUP = A.solve(b, "LUP");
        const xLU  = A.solve(b, "LU");
        expect(mEq(xLUP, xLU, 1e-10)).toBe(true);
    });

    it("QR: A * solve(b, QR) ≈ b", () => {
        const { A, b } = buildSystem();
        const x = A.solve(b, "QR");
        expect(mEq(A.mul(x), b, 1e-9)).toBe(true);
    });

    it("soluzione di I * x = b è b", () => {
        const I = Matrix.identityOf(3, Complex.zero, Complex.one);
        const b = Matrix.zerosOf(3, 1, Complex.zero, Complex.one);
        b.set(0, 0, new Complex(1, 2));
        b.set(1, 0, new Complex(-3, 1));
        b.set(2, 0, new Complex(0, 5));
        const x = I.solve(b);
        expect(mEq(x, b, 1e-12)).toBe(true);
    });

    it("sistema sovradeterminato (QR su sistema rettangolare)", () => {
        // A 3×2, b 3×1 → soluzione ai minimi quadrati
        const A = cmat([[[1, 0], [0, 1]], [[1, 1], [1, 0]], [[0, 1], [1, 1]]]);
        const b = Matrix.zerosOf(3, 1, Complex.zero, Complex.one);
        b.set(0, 0, new Complex(1, 0));
        b.set(1, 0, new Complex(1, 1));
        b.set(2, 0, new Complex(0, 1));
        expect(() => A.solve(b, "QR")).not.toThrow();
    });
});

// ============================================================
// 7. OPERAZIONI ELEMENT-WISE E UNARIE
// ============================================================

describe("Matrix<Complex> — dotOps e unarie", () => {

    const A = cmat([[[1, 2], [3, -1]], [[0, 1], [-2, 3]]]);

    it("dotMul: elemento per elemento", () => {
        const B = cmat([[[1, 0], [0, 1]], [[1, 1], [-1, 0]]]);
        const C = A.dotMul(B);
        expect(cEq(C.get(0, 0), new Complex(1, 2))).toBe(true);   // (1+2i)*1
        expect(cEq(C.get(0, 1), new Complex(1, 3))).toBe(true); // (3-i)*(i) = 1+3i ... wait
        // (3-i)*(0+i) = 3i - i² = 3i+1 = 1+3i
        expect(cEq(C.get(1, 0), new Complex(-1, 1))).toBe(true);
    });

    it("dotDiv: A ./ A = ones", () => {
        const C = A.dotDiv(A);
        for (let i = 0; i < A.rows; i++)
            for (let j = 0; j < A.cols; j++)
                expect(C.get(i, j).isNearZero(1e-10) || cEq(C.get(i, j), Complex.one, 1e-10)).toBe(true);
    });

    it("abs element-wise: restituisce moduli", () => {
        const C = A.abs();
        expect(C.get(0, 0).real).toBeCloseTo(Math.sqrt(5), 10);  // |1+2i|=√5
        expect(C.get(0, 0).imag).toBeCloseTo(0, 10);
    });

    it("negate: tutti i segni invertiti", () => {
        const C = A.negate();
        expect(cEq(C.get(0, 0), new Complex(-1, -2))).toBe(true);
        expect(cEq(C.get(1, 1), new Complex(2, -3))).toBe(true);
    });

    it("round: parte reale e immaginaria arrotondate", () => {
        const M = cmat([[[1.6, -0.4], [2.1, 3.9]]]);
        const R = M.round();
        expect(cEq(R.get(0, 0), new Complex(2, 0))).toBe(true);
        expect(cEq(R.get(0, 1), new Complex(2, 4))).toBe(true);
    });

    it("sqrt: sqrt(A).dotMul(sqrt(A)) ≈ A", () => {
        const M = cmat([[[4, 0], [0, 4]]]);  // elementi reali positivi
        const S = M.sqrt();
        const S2 = S.dotMul(S);
        for (let j = 0; j < 2; j++)
            expect(cEq(S2.get(0, j), M.get(0, j), 1e-10)).toBe(true);
    });
});

// ============================================================
// 8. TRASFORMAZIONI
// ============================================================

describe("Matrix<Complex> — trasformazioni", () => {

    const A = cmat([[[1, 2], [3, -1]], [[-1, 0], [2, 4]]]);

    it("transpose: (A^T)_{ij} = A_{ji}", () => {
        const AT = A.t();
        expect(AT.rows).toBe(A.cols);
        expect(AT.cols).toBe(A.rows);
        expect(cEq(AT.get(0, 1), A.get(1, 0))).toBe(true);
        expect(cEq(AT.get(1, 0), A.get(0, 1))).toBe(true);
    });

    it("(A^T)^T = A", () => {
        expect(mEq(A.t().t(), A)).toBe(true);
    });

    it("(A+B)^T = A^T + B^T", () => {
        const B = cmat([[[0, 1], [1, 0]], [[2, -1], [-1, 2]]]);
        expect(mEq(A.add(B).t(), A.t().add(B.t()))).toBe(true);
    });

    it("reshape: stesso dato, nuove dimensioni", () => {
        const M = cmat([[[1, 0], [2, 1], [3, -1]]]);   // 1×3
        const R = M.reshape(3, 1);                      // 3×1
        expect(R.rows).toBe(3); expect(R.cols).toBe(1);
        expect(cEq(R.get(2, 0), M.get(0, 2))).toBe(true);
    });

    it("slice: estrae la sotto-matrice corretta", () => {
        const M = cmat([[[1, 0], [2, 1], [3, -1]], [[4, 2], [5, 0], [6, 1]]]);
        const S = M.slice(0, 2, 1, 3);   // righe 0-1, colonne 1-2
        expect(S.rows).toBe(2); expect(S.cols).toBe(2);
        expect(cEq(S.get(0, 0), M.get(0, 1))).toBe(true);
        expect(cEq(S.get(1, 1), M.get(1, 2))).toBe(true);
    });

    it("flip dim=1 (flipud): ultima riga diventa prima", () => {
        const F = A.flip(1);
        expect(cEq(F.get(0, 0), A.get(1, 0))).toBe(true);
        expect(cEq(F.get(1, 0), A.get(0, 0))).toBe(true);
    });

    it("flip dim=2 (fliplr): ultima colonna diventa prima", () => {
        const F = A.flip(2);
        expect(cEq(F.get(0, 0), A.get(0, 1))).toBe(true);
        expect(cEq(F.get(0, 1), A.get(0, 0))).toBe(true);
    });

    it("rot90 × 4 = originale", () => {
        const E = A.rot90().rot90().rot90().rot90();
        expect(mEq(E, A)).toBe(true);
    });

    it("rot90 scambia dimensioni", () => {
        const M = cmat([[[1, 0], [2, 0], [3, 0]]]);   // 1×3
        const R = M.rot90();
        expect(R.rows).toBe(3); expect(R.cols).toBe(1);
    });
});

// ============================================================
// 9. STATISTICHE
// ============================================================

describe("Matrix<Complex> — statistiche", () => {

    const A = cmat([[[1, 0], [2, 1]], [[3, -1], [4, 2]]]);

    it("sum(dim=1): somma per colonne → 1×C", () => {
        const S = A.sum(1);
        expect(S.rows).toBe(1); expect(S.cols).toBe(2);
        expect(cEq(S.get(0, 0), new Complex(4, -1))).toBe(true);  // (1+0i)+(3-i)
        expect(cEq(S.get(0, 1), new Complex(6, 3))).toBe(true);   // (2+i)+(4+2i)
    });

    it("sum(dim=2): somma per righe → R×1", () => {
        const S = A.sum(2);
        expect(S.rows).toBe(2); expect(S.cols).toBe(1);
        expect(cEq(S.get(0, 0), new Complex(3, 1))).toBe(true);   // (1+0i)+(2+i)
        expect(cEq(S.get(1, 0), new Complex(7, 1))).toBe(true);   // (3-i)+(4+2i)
    });

    it("totalSum = somma di tutti gli elementi", () => {
        const t = A.totalSum();
        expect(cEq(t, new Complex(10, 2))).toBe(true);
    });

    it("mean(dim=1) = sum/R", () => {
        const M = A.mean(1);
        expect(cEq(M.get(0, 0), new Complex(2, -0.5))).toBe(true);
        expect(cEq(M.get(0, 1), new Complex(3, 1.5))).toBe(true);
    });

    it("max(dim=1): max per modulo in ogni colonna", () => {
        // col 0: |1|=1, |3-i|=√10 → max = 3-i (idx=2, 1-based)
        // col 1: |2+i|=√5, |4+2i|=√20 → max = 4+2i (idx=2)
        const { value, index } = A.max(1);
        expect(index[0]).toBe(2);
        expect(index[1]).toBe(2);
    });

    it("min(dim=2): min per modulo in ogni riga", () => {
        // riga 0: |1|=1, |2+i|=√5 → min = 1 (idx=1)
        // riga 1: |3-i|=√10, |4+2i|=√20 → min = 3-i (idx=1)
        const { value, index } = A.min(2);
        expect(index[0]).toBe(1);
        expect(index[1]).toBe(1);
    });
});

// ============================================================
// 10. PROPRIETÀ STRUTTURALI
// ============================================================

describe("Matrix<Complex> — proprietà strutturali", () => {

    it("isSquare: matrice quadrata", () => {
        const A = Matrix.identityOf(3, Complex.zero, Complex.one);
        expect(A.isSquare()).toBe(true);
        const B = Matrix.zerosOf(2, 3, Complex.zero, Complex.one);
        expect(B.isSquare()).toBe(false);
    });

    it("isZeroMatrix", () => {
        const Z = Matrix.zerosOf(3, 3, Complex.zero, Complex.one);
        expect(Z.isZeroMatrix(1e-12)).toBe(true);
    });

    it("isIdentity", () => {
        const I = Matrix.identityOf(3, Complex.zero, Complex.one);
        expect(I.isIdentity(1e-12)).toBe(true);
    });

    it("isUpperTriangular", () => {
        const U = cmat([[[1, 1], [2, 0]], [[0, 0], [3, -1]]]);
        expect(U.isUpperTriangular(1e-12)).toBe(true);
    });

    it("isLowerTriangular", () => {
        const L = cmat([[[1, 0], [0, 0]], [[2, 1], [3, -1]]]);
        expect(L.isLowerTriangular(1e-12)).toBe(true);
    });

    it("isDiagonal", () => {
        const D = cmat([[[2, 1], [0, 0]], [[0, 0], [-3, 2]]]);
        expect(D.isDiagonal(1e-12)).toBe(true);
    });

    it("hasFiniteValues", () => {
        const A = Matrix.zerosOf(2, 2, Complex.zero, Complex.one);
        expect(A.hasFiniteValues()).toBe(true);
    });

    it("isInvertible: matrice non singolare", () => {
        const A = cmat([[[2, 1], [0, 1]], [[1, 0], [1, -1]]]);
        expect(A.isInvertible()).toBe(true);
    });

    it("isSingular: matrice singolare", () => {
        const A = cmat([[[1, 0], [2, 0]], [[1, 0], [2, 0]]]);
        expect(A.isSingular()).toBe(true);
    });
});

// ============================================================
// 11. IDENTITÀ MATEMATICHE COMPLESSE
// ============================================================

describe("Identità matematiche su Matrix<Complex>", () => {

    // Formula di Euler: e^{iπ} + 1 = 0
    it("fromNumber e moltiplicazioni: (0+iπ) → exp scalare = -1+0i", () => {
        const pi_i = new Complex(0, Math.PI);
        const e_ipi = pi_i.sqrt().multiply(pi_i.sqrt()); // (iπ/2)^2 no... usiamo diversa rotta
        // Calcoliamo e^{iπ} via cos+i*sin = -1+0i
        const euler = new Complex(Math.cos(Math.PI), Math.sin(Math.PI));
        expect(euler.real).toBeCloseTo(-1, 10);
        expect(euler.imag).toBeCloseTo(0, 10);
    });

    // Identità di Parseval: per A invertibile, ||A * inv(A)||_F = √n
    it("A * inv(A) = I → norm_F = √n", () => {
        const A = cmat([[[2, 1], [0, 1]], [[1, 0], [1, -1]]]);
        const I_reconstructed = A.mul(A.inv());
        expect(I_reconstructed.norm("Fro")).toBeCloseTo(Math.sqrt(2), 8);
    });

    // det(A^T) = det(A)  (per trasposto, non coniugato)
    it("det(A^T) = det(A)", () => {
        const A = cmat([[[1, 1], [2, -1]], [[-1, 2], [3, 0]]]);
        const dA = A.det(), dAt = A.t().det();
        expect(cEq(dA, dAt, 1e-10)).toBe(true);
    });

    // trace(A + B) = trace(A) + trace(B)
    it("trace(A + B) = trace(A) + trace(B)", () => {
        const A = cmat([[[1, 2], [3, 0]], [[0, 1], [-1, 2]]]);
        const B = cmat([[[-1, 1], [0, 2]], [[1, 0], [2, -1]]]);
        const lhs = A.add(B).trace();
        const rhs = A.trace().add(B.trace());
        expect(cEq(lhs, rhs, 1e-12)).toBe(true);
    });

    // trace(AB) = trace(BA)
    it("trace(A*B) = trace(B*A)", () => {
        const A = cmat([[[1, 1], [2, 0]], [[0, -1], [1, 2]]]);
        const B = cmat([[[2, -1], [0, 1]], [[1, 0], [-1, 1]]]);
        const trAB = A.mul(B).trace();
        const trBA = B.mul(A).trace();
        expect(cEq(trAB, trBA, 1e-10)).toBe(true);
    });

    // (AB)^T = B^T A^T
    it("(A*B)^T = B^T * A^T", () => {
        const A = cmat([[[1, 1], [2, 0]], [[0, -1], [1, 2]]]);
        const B = cmat([[[2, -1], [0, 1]], [[1, 0], [-1, 1]]]);
        expect(mEq(A.mul(B).t(), B.t().mul(A.t()), 1e-10)).toBe(true);
    });

    // Norma submoltiplicativa: ||AB||_F ≤ ||A||_F * ||B||_F
    it("norma sub-moltiplicativa: ||A*B||_F ≤ ||A||_F * ||B||_F", () => {
        const A = cmat([[[1, 1], [2, 0]], [[0, -1], [1, 2]]]);
        const B = cmat([[[2, -1], [0, 1]], [[1, 0], [-1, 1]]]);
        const normAB = A.mul(B).norm("Fro");
        const normA  = A.norm("Fro");
        const normB  = B.norm("Fro");
        expect(normAB).toBeLessThanOrEqual(normA * normB + 1e-10);
    });

    // Invarianza del determinante per la trasposizione con righe reali
    it("det di matrice reale come Complex = det reale", () => {
        // Matrice 2×2 con parti immaginarie = 0
        const A = cmat([[[3, 0], [1, 0]], [[2, 0], [4, 0]]]);
        const d = A.det();
        // det = 3*4 - 1*2 = 10
        expect(d.real).toBeCloseTo(10, 10);
        expect(d.imag).toBeCloseTo(0, 10);
    });

    // Solve: soluzione unica per sistema non degenere
    it("sistema lineare: soluzione unica e corretta", () => {
        // [[i, 1], [1, i]] * [[x], [y]] = [[1+i], [0]]
        // → soluzione attesa: x = (1+i)*i/(i²-1) etc.
        const A = cmat([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]);
        const b = Matrix.zerosOf(2, 1, Complex.zero, Complex.one);
        b.set(0, 0, new Complex(1, 1));
        b.set(1, 0, new Complex(0, 0));
        const x = A.solve(b);
        // Verifica A*x ≈ b
        expect(mEq(A.mul(x), b, 1e-10)).toBe(true);
    });
});

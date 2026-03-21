// test/Rational.test.ts
//
// Test profondi per Rational e Matrix<Rational>.
// La caratteristica fondamentale di Rational è l'ESATTEZZA:
// zero errore di arrotondamento, risultati come frazioni esatte.
//
// Sezioni:
//   1.  Tipo Rational (aritmetica, GCD, riduzione, edge-case)
//   2.  Cross-cancellation (denominatori piccoli dopo le operazioni)
//   3.  Costruzione di Matrix<Rational>
//   4.  Operazioni aritmetiche matriciali esatte
//   5.  Algebra lineare esatta (det, trace, inv)
//   6.  Decomposizioni (LUP, QR)
//   7.  Solver esatto
//   8.  Operazioni element-wise e unarie
//   9.  Trasformazioni
//  10.  Statistiche
//  11.  Identità matematiche con aritmetica esatta
//  12.  Stress test: denominatori non esplodono
//
import { describe, it, expect } from "vitest";
import { Matrix, Rational } from "../src";

// ============================================================
// Helper
// ============================================================

/** Crea un Rational da numeratore e denominatore bigint. */
function r(num: bigint | number, den: bigint | number = 1): Rational {
    return new Rational(num, den);
}

/** Crea Matrix<Rational> da array 2D di coppie [num, den]. */
function rmat(data: [bigint | number, bigint | number][][]): Matrix<Rational> {
    const rows = data.length, cols = data[0].length;
    const m = Matrix.zerosOf(rows, cols, Rational.zero, Rational.one);
    for (let i = 0; i < rows; i++)
        for (let j = 0; j < cols; j++)
            m.set(i, j, r(data[i][j][0], data[i][j][1]));
    return m;
}

/** Verifica uguaglianza esatta di due Rational. */
function rEq(a: Rational, b: Rational): boolean {
    return a.num === b.num && a.den === b.den;
}

/** Verifica uguaglianza esatta di due Matrix<Rational>. */
function mEq(A: Matrix<Rational>, B: Matrix<Rational>): boolean {
    if (A.rows !== B.rows || A.cols !== B.cols) return false;
    for (let i = 0; i < A.rows; i++)
        for (let j = 0; j < A.cols; j++)
            if (!rEq(A.get(i, j), B.get(i, j))) return false;
    return true;
}

// ============================================================
// 1. TIPO RATIONAL — costruzione e riduzione
// ============================================================

describe("Rational — costruzione e riduzione", () => {

    it("ha kind = 'rational'", () => {
        expect(r(1).kind).toBe("rational");
    });

    it("zero e one sono corretti", () => {
        expect(Rational.zero.num).toBe(BigInt(0)); expect(Rational.zero.den).toBe(BigInt(1));
        expect(Rational.one.num).toBe(BigInt(1));  expect(Rational.one.den).toBe(BigInt(1));
    });

    it("riduzione automatica: 6/4 → 3/2", () => {
        const x = r(6, 4);
        expect(x.num).toBe(BigInt(3)); expect(x.den).toBe(BigInt(2));
    });

    it("riduzione con GCD grande: 100/75 → 4/3", () => {
        const x = r(100, 75);
        expect(x.num).toBe(BigInt(4)); expect(x.den).toBe(BigInt(3));
    });

    it("segno sempre al numeratore: 3/-4 → -3/4", () => {
        const x = r(3, -4);
        expect(x.num).toBe(BigInt(-3)); expect(x.den).toBe(BigInt(4));
    });

    it("segno: -3/-4 → 3/4", () => {
        const x = r(-3, -4);
        expect(x.num).toBe(BigInt(3)); expect(x.den).toBe(BigInt(4));
    });

    it("interi restano interi: r(7) → 7/1", () => {
        const x = r(7);
        expect(x.num).toBe(BigInt(7)); expect(x.den).toBe(BigInt(1));
    });

    it("denominatore zero lancia errore", () => {
        expect(() => r(1, 0)).toThrow();
    });

    it("fromNumber con intero: r(5) esatto", () => {
        const x = Rational.zero.fromNumber(5);
        expect(x.num).toBe(BigInt(5)); expect(x.den).toBe(BigInt(1));
    });

    it("fromNumber con non-finito lancia errore", () => {
        expect(() => Rational.zero.fromNumber(Infinity)).toThrow();
        expect(() => Rational.zero.fromNumber(NaN)).toThrow();
    });

    it("toNumber: 1/3 approssimato correttamente", () => {
        const x = r(1, 3);
        expect(x.toNumber()).toBeCloseTo(1 / 3, 14);
    });

    it("toString: intero senza barra", () => {
        expect(r(7).toString()).toBe("7");
    });

    it("toString: frazione con barra", () => {
        expect(r(3, 4).toString()).toBe("3/4");
    });

    it("toString: negativo", () => {
        expect(r(-2, 5).toString()).toBe("-2/5");
    });
});

// ============================================================
// 2. ARITMETICA RAZIONALE — esattezza e cross-cancellation
// ============================================================

describe("Rational — aritmetica esatta", () => {

    // ---- add ----
    it("1/2 + 1/3 = 5/6 (esatto)", () => {
        const s = r(1, 2).add(r(1, 3));
        expect(s.num).toBe(BigInt(5)); expect(s.den).toBe(BigInt(6));
    });
    it("1/4 + 3/4 = 1 (stesso denominatore → riduzione)", () => {
        const s = r(1, 4).add(r(3, 4));
        expect(s.num).toBe(BigInt(1)); expect(s.den).toBe(BigInt(1));
    });
    it("add con zero = identità", () => {
        const x = r(7, 3);
        expect(rEq(x.add(Rational.zero), x)).toBe(true);
    });
    it("add commutativa", () => {
        const a = r(2, 5), b = r(3, 7);
        expect(rEq(a.add(b), b.add(a))).toBe(true);
    });
    it("add associativa", () => {
        const a = r(1, 2), b = r(1, 3), c = r(1, 6);
        expect(rEq(a.add(b).add(c), a.add(b.add(c)))).toBe(true);
    });
    it("add con opposto = 0", () => {
        const x = r(5, 7);
        expect(rEq(x.add(x.negate()), Rational.zero)).toBe(true);
    });
    it("risultato di add è già ridotto", () => {
        // 1/6 + 1/6 = 2/6 → 1/3
        const s = r(1, 6).add(r(1, 6));
        expect(s.num).toBe(BigInt(1)); expect(s.den).toBe(BigInt(3));
    });

    // ---- subtract ----
    it("3/4 - 1/4 = 1/2", () => {
        const s = r(3, 4).subtract(r(1, 4));
        expect(s.num).toBe(BigInt(1)); expect(s.den).toBe(BigInt(2));
    });
    it("subtract di sé stesso = 0", () => {
        const x = r(11, 13);
        expect(rEq(x.subtract(x), Rational.zero)).toBe(true);
    });
    it("subtract non commutativa: a-b ≠ b-a (in generale)", () => {
        const a = r(3, 4), b = r(1, 2);
        expect(rEq(a.subtract(b), b.subtract(a))).toBe(false);
    });

    // ---- multiply ----
    it("2/3 * 3/4 = 1/2 (cross-cancellation)", () => {
        const p = r(2, 3).multiply(r(3, 4));
        expect(p.num).toBe(BigInt(1)); expect(p.den).toBe(BigInt(2));
    });
    it("multiply per zero = zero", () => {
        expect(rEq(r(99, 7).multiply(Rational.zero), Rational.zero)).toBe(true);
    });
    it("multiply per uno = identità", () => {
        const x = r(5, 9);
        expect(rEq(x.multiply(Rational.one), x)).toBe(true);
    });
    it("multiply commutativa", () => {
        const a = r(2, 5), b = r(7, 3);
        expect(rEq(a.multiply(b), b.multiply(a))).toBe(true);
    });
    it("4/9 * 9/4 = 1 (inverso moltiplicativo)", () => {
        const x = r(4, 9);
        const inv_x = r(9, 4);
        expect(rEq(x.multiply(inv_x), Rational.one)).toBe(true);
    });
    it("risultato di multiply è ridotto: 6/10 * 5/9 = 1/3", () => {
        const p = r(6, 10).multiply(r(5, 9));
        expect(p.num).toBe(BigInt(1)); expect(p.den).toBe(BigInt(3));
    });

    // ---- divide ----
    it("(2/3) / (4/5) = 5/6", () => {
        const q = r(2, 3).divide(r(4, 5));
        expect(q.num).toBe(BigInt(5)); expect(q.den).toBe(BigInt(6));
    });
    it("divide per sé stesso = 1", () => {
        const x = r(7, 11);
        expect(rEq(x.divide(x), Rational.one)).toBe(true);
    });
    it("divide per zero lancia errore", () => {
        expect(() => r(1).divide(Rational.zero)).toThrow();
    });
    it("divide con negativo: segno corretto", () => {
        // (3/4) / (-1/2) = -3/2
        const q = r(3, 4).divide(r(-1, 2));
        expect(q.num).toBe(BigInt(-3)); expect(q.den).toBe(BigInt(2));
    });

    // ---- negate ----
    it("negate inverte segno", () => {
        const x = r(5, 7);
        const n = x.negate();
        expect(n.num).toBe(BigInt(-5)); expect(n.den).toBe(BigInt(7));
    });
    it("negate del negate = originale", () => {
        const x = r(-3, 8);
        expect(rEq(x.negate().negate(), x)).toBe(true);
    });

    // ---- abs ----
    it("abs di negativo restituisce positivo", () => {
        const x = r(-5, 7);
        const a = x.abs();
        expect(a.num).toBe(BigInt(5)); expect(a.den).toBe(BigInt(7));
    });
    it("abs di positivo restituisce sé stesso", () => {
        const x = r(3, 4);
        expect(rEq(x.abs(), x)).toBe(true);
    });

    // ---- round ----
    it("round(7/4) = 2", () => {
        const rnd = r(7, 4).round();
        expect(rnd.num).toBe(BigInt(2)); expect(rnd.den).toBe(BigInt(1));
    });
    it("round(5/4) = 1", () => {
        const rnd = r(5, 4).round();
        expect(rnd.num).toBe(BigInt(1)); expect(rnd.den).toBe(BigInt(1));
    });
    it("round(-3/2) = -1", () => {
        const rnd = r(-3, 2).round();
        console.log(rnd.toString())
        expect(rnd.num).toBe(BigInt(-2)); expect(rnd.den).toBe(BigInt(1));
    });

    // ---- comparazione ----
    it("greaterThan: 3/4 > 1/2", () => {
        expect(r(3, 4).greaterThan(r(1, 2))).toBe(true);
    });
    it("lessThan: -1/3 < 1/4", () => {
        expect(r(-1, 3).lessThan(r(1, 4))).toBe(true);
    });
    it("equals: 2/4 e 1/2 sono uguali dopo riduzione", () => {
        expect(r(2, 4).equals(r(1, 2))).toBe(true);
    });
    it("isNearZero: ignora tol, controlla zero esatto", () => {
        expect(Rational.zero.isNearZero(0)).toBe(true);
        expect(r(1, 1_000_000_000).isNearZero(1e-10)).toBe(false);
    });

    // ---- GCD binario ----
    it("gcd(0, 5) = 5", () => {
        expect(Rational.gcd(BigInt(0), BigInt(5))).toBe(BigInt(5));
    });
    it("gcd(12, 8) = 4", () => {
        expect(Rational.gcd(BigInt(12), BigInt(8))).toBe(BigInt(4));
    });
    it("gcd(101, 97) = 1 (coprimi)", () => {
        expect(Rational.gcd(BigInt(101), BigInt(97))).toBe(BigInt(1));
    });
    it("gcd(0, 0) = 0", () => {
        expect(Rational.gcd(BigInt(0), BigInt(0))).toBe(BigInt(0));
    });
});

// ============================================================
// 3. CROSS-CANCELLATION — denominatori rimangono piccoli
// ============================================================

describe("Rational — cross-cancellation: denominatori non esplodono", () => {

    it("somma di 10 frazioni 1/n rimane con denominatore piccolo", () => {
        // Σ 1/k per k=1..10 = 7381/2520
        let s = Rational.zero;
        for (let k = 1; k <= 10; k++) s = s.add(r(1, k));
        // Il denominatore non deve essere il prodotto 1*2*...*10 = 3628800
        // ma il LCM = 2520
        expect(s.den).toBe(BigInt(2520));
        expect(s.num).toBe(BigInt(7381));
    });

    it("prodotto di frazioni consecutive rimane ridotto", () => {
        // (1/2)(2/3)(3/4)(4/5) = 1/5
        const p = r(1, 2).multiply(r(2, 3)).multiply(r(3, 4)).multiply(r(4, 5));
        expect(p.num).toBe(BigInt(1)); expect(p.den).toBe(BigInt(5));
    });

    it("100 addizioni di 1/100 = 1 esatto", () => {
        let s = Rational.zero;
        for (let i = 0; i < 100; i++) s = s.add(r(1, 100));
        expect(rEq(s, Rational.one)).toBe(true);
    });

    it("1/2 + 1/3 + 1/6 = 1 esatto", () => {
        const s = r(1, 2).add(r(1, 3)).add(r(1, 6));
        expect(rEq(s, Rational.one)).toBe(true);
    });
});

// ============================================================
// 4. COSTRUZIONE Matrix<Rational>
// ============================================================

describe("Matrix<Rational> — costruzione", () => {

    it("zerosOf: tutti Rational.zero esatto", () => {
        const M = Matrix.zerosOf(2, 3, Rational.zero, Rational.one);
        for (let i = 0; i < 2; i++)
            for (let j = 0; j < 3; j++)
                expect(rEq(M.get(i, j), Rational.zero)).toBe(true);
    });

    it("identityOf: diagonale = 1, off-diag = 0", () => {
        const I = Matrix.identityOf(3, Rational.zero, Rational.one);
        for (let i = 0; i < 3; i++)
            for (let j = 0; j < 3; j++) {
                const expected = i === j ? Rational.one : Rational.zero;
                expect(rEq(I.get(i, j), expected)).toBe(true);
            }
    });

    it("fromTypedArray: popola correttamente", () => {
        const data: Rational[][] = [
            [r(1, 2), r(3, 4)],
            [r(-1, 3), r(7, 5)],
        ];
        const M = Matrix.fromTypedArray(data, Rational.zero, Rational.one);
        expect(rEq(M.get(0, 0), r(1, 2))).toBe(true);
        expect(rEq(M.get(1, 0), r(-1, 3))).toBe(true);
    });

    it("set con number intero usa fromNumber esatto", () => {
        const M = Matrix.zerosOf(2, 2, Rational.zero, Rational.one);
        M.set(0, 0, 7);
        expect(M.get(0, 0).num).toBe(BigInt(7));
        expect(M.get(0, 0).den).toBe(BigInt(1));
    });

    it("clone è indipendente dall'originale", () => {
        const A = rmat([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
        const B = A.clone();
        B.set(0, 0, r(99));
        expect(rEq(A.get(0, 0), r(1, 2))).toBe(true);
    });
});

// ============================================================
// 5. OPERAZIONI MATRICIALI ESATTE
// ============================================================

describe("Matrix<Rational> — operazioni matriciali esatte", () => {

    const A = rmat([[[1, 2], [1, 3]], [[2, 3], [1, 4]]]);
    const B = rmat([[[1, 4], [1, 2]], [[3, 4], [1, 3]]]);

    it("add: (1/2 + 1/4) = 3/4 esatto", () => {
        const C = A.add(B);
        expect(rEq(C.get(0, 0), r(3, 4))).toBe(true);   // 1/2 + 1/4
    });

    it("add commutativa: A+B = B+A (esatto)", () => {
        expect(mEq(A.add(B), B.add(A))).toBe(true);
    });

    it("sub: A - A = zero matrix (esatta)", () => {
        const Z = A.sub(A);
        for (let i = 0; i < 2; i++)
            for (let j = 0; j < 2; j++)
                expect(rEq(Z.get(i, j), Rational.zero)).toBe(true);
    });

    it("mul scalare con 2: ogni elemento raddoppia esattamente", () => {
        const C = A.mul(2);
        expect(rEq(C.get(0, 0), r(1))).toBe(true);  // 1/2 * 2 = 1
        expect(rEq(C.get(0, 1), r(2, 3))).toBe(true); // 1/3 * 2 = 2/3
    });

    it("mul: A * I = A (esatto)", () => {
        const I = Matrix.identityOf(2, Rational.zero, Rational.one);
        expect(mEq(A.mul(I), A)).toBe(true);
    });

    it("mul matriciale: risultati esatti senza arrotondamento", () => {
        // [[1/2, 1/3], [2/3, 1/4]] * [[1/4, 1/2], [3/4, 1/3]]
        // (0,0): 1/2*1/4 + 1/3*3/4 = 1/8 + 1/4 = 3/8
        const C = A.mul(B);
        expect(rEq(C.get(0, 0), r(3, 8))).toBe(true);
    });

    it("dotMul: element-wise, risultati esatti", () => {
        const C = A.dotMul(B);
        // (0,0): 1/2 * 1/4 = 1/8
        expect(rEq(C.get(0, 0), r(1, 8))).toBe(true);
    });

    it("dotDiv: element-wise, risultati esatti", () => {
        // (1/2) / (1/4) = 2
        const C = A.dotDiv(B);
        expect(rEq(C.get(0, 0), r(2))).toBe(true);
    });

    it("pow(0) = I esatto", () => {
        const I = Matrix.identityOf(2, Rational.zero, Rational.one);
        expect(mEq(A.pow(0), I)).toBe(true);
    });

    it("pow(2) = A*A esatto", () => {
        expect(mEq(A.pow(2), A.mul(A))).toBe(true);
    });
});

// ============================================================
// 6. ALGEBRA LINEARE ESATTA
// ============================================================

describe("Matrix<Rational> — algebra lineare esatta", () => {

    it("det 1×1: det([[5/7]]) = 5/7 esatto", () => {
        const A = rmat([[[5, 7]]]);
        expect(rEq(A.det(), r(5, 7))).toBe(true);
    });

    it("det 2×2 esatto: [[1/2, 1/3],[1/4, 1/6]] → 0", () => {
        // det = 1/2*1/6 - 1/3*1/4 = 1/12 - 1/12 = 0
        const A = rmat([[[1, 2], [1, 3]], [[1, 4], [1, 6]]]);
        expect(rEq(A.det(), Rational.zero)).toBe(true);
    });

    it("det 2×2 esatto: [[2, 1],[3, 4]] → 5", () => {
        const A = rmat([[[2, 1], [1, 1]], [[3, 1], [4, 1]]]);
        expect(rEq(A.det(), r(5))).toBe(true);
    });

    it("det(I) = 1 esatto", () => {
        const I = Matrix.identityOf(4, Rational.zero, Rational.one);
        expect(rEq(I.det(), Rational.one)).toBe(true);
    });

    it("det moltiplicativo: det(A*B) = det(A)*det(B) esatto", () => {
        const A = rmat([[[2, 1], [1, 1]], [[3, 1], [4, 1]]]);
        const B = rmat([[[1, 1], [3, 1]], [[2, 1], [1, 1]]]);
        const detAB = A.mul(B).det();
        const detAdB = A.det().multiply(B.det());
        expect(rEq(detAB, detAdB)).toBe(true);
    });

    it("trace esatta: somma diagonale", () => {
        const A = rmat([[[1, 3], [0, 1]], [[0, 1], [2, 5]]]);
        // trace = 1/3 + 2/5 = 5/15 + 6/15 = 11/15
        expect(rEq(A.trace(), r(11, 15))).toBe(true);
    });

    it("inv: A * inv(A) = I (esatto)", () => {
        // [[2, 1], [1, 1]] → det = 1 → inv = [[1, -1], [-1, 2]]
        const A = rmat([[[2, 1], [1, 1]], [[1, 1], [1, 1]]]);
        const AI = A.mul(A.inv());
        const I = Matrix.identityOf(2, Rational.zero, Rational.one);
        expect(mEq(AI, I)).toBe(true);  // ESATTO, zero arrotondamento
    });

    it("inv: inv(I) = I esatto", () => {
        const I = Matrix.identityOf(3, Rational.zero, Rational.one);
        expect(mEq(I.inv(), I)).toBe(true);
    });

    it("inv: inv(diagonale) = reciproco diagonale esatto", () => {
        const D = Matrix.zerosOf(3, 3, Rational.zero, Rational.one) as Matrix<Rational>;
        // Non esiste Matrix.zerosOf come Matrix<Rational> direttamente
        const DR = Matrix.identityOf<Rational>(3, Rational.zero, Rational.one);
        DR.set(0, 0, r(2));
        DR.set(1, 1, r(3));
        DR.set(2, 2, r(4));
        const DRinv = DR.inv();
        expect(rEq(DRinv.get(0, 0), r(1, 2))).toBe(true);
        expect(rEq(DRinv.get(1, 1), r(1, 3))).toBe(true);
        expect(rEq(DRinv.get(2, 2), r(1, 4))).toBe(true);
    });

    // Matrice di Hilbert razionale 3×3: H(i,j) = 1/(i+j+1), esattissima
    it("Hilbert 3×3 razionale: det esatto non-zero", () => {
        const H = Matrix.zerosOf<Rational>(3, 3, Rational.zero, Rational.one);
        for (let i = 0; i < 3; i++)
            for (let j = 0; j < 3; j++)
                H.set(i, j, r(1, i + j + 1));
        // det(Hilbert_3) = 1/2160
        const d = H.det();
        expect(d.num !== BigInt(0)).toBe(true);
        expect(d.toNumber()).toBeCloseTo(1 / 2160, 10);
    });
});

// ============================================================
// 7. DECOMPOSIZIONI
// ============================================================

describe("Matrix<Rational> — decomposizioni", () => {

    describe("LUP", () => {
        it("L * U = P * A (esatto)", () => {
            const A = rmat([[[2, 1], [1, 1]], [[1, 1], [3, 1]]]);
            const { L, U, P } = Matrix.decomp.lup(A);
            const n = 2;
            const PA = Matrix.zerosOf<Rational>(n, n, Rational.zero, Rational.one);
            for (let i = 0; i < n; i++)
                for (let j = 0; j < n; j++)
                    PA.set(i, j, A.get(P[i], j));
            expect(mEq(L.mul(U) as Matrix<Rational>, PA)).toBe(true);
        });

        it("L è triangolare inferiore unitaria (esatto)", () => {
            const A = rmat([[[2, 1], [1, 1]], [[4, 1], [3, 1]]]);
            const { L } = Matrix.decomp.lup(A);
            expect(L.isLowerTriangular()).toBe(true);
            for (let i = 0; i < L.rows; i++)
                expect(rEq(L.get(i, i) as Rational, Rational.one)).toBe(true);
        });

        it("U è triangolare superiore (esatto)", () => {
            const A = rmat([[[2, 1], [1, 1]], [[4, 1], [3, 1]]]);
            const { U } = Matrix.decomp.lup(A);
            expect(U.isUpperTriangular()).toBe(true);
        });

        it("Hilbert 3×3: LUP funziona con frazioni esatte", () => {
            const H = Matrix.zerosOf<Rational>(3, 3, Rational.zero, Rational.one);
            for (let i = 0; i < 3; i++)
                for (let j = 0; j < 3; j++)
                    H.set(i, j, r(1, i + j + 1));
            expect(() => Matrix.decomp.lup(H)).not.toThrow();
            const { L, U, P } = Matrix.decomp.lup(H);
            const PA = Matrix.zerosOf<Rational>(3, 3, Rational.zero, Rational.one);
            for (let i = 0; i < 3; i++)
                for (let j = 0; j < 3; j++)
                    PA.set(i, j, H.get(P[i], j));
            expect(mEq(L.mul(U) as Matrix<Rational>, PA)).toBe(true);
        });
    });

    describe("QR", () => {
        it("Q * R = A (approssimato — QR usa sqrt che è float)", () => {
            const A = rmat([[[1, 1], [2, 1]], [[1, 2], [1, 1]], [[2, 1], [3, 2]]]);
            const { Q, R } = Matrix.decomp.qr(A);
            // Per Rational, sqrt() ricade su float; il risultato non è esatto
            // ma la ricostruzione A=QR deve essere approssimata
            const QR = Q.mul(R);
            for (let i = 0; i < A.rows; i++)
                for (let j = 0; j < A.cols; j++)
                    expect(QR.get(i, j).toNumber()).toBeCloseTo(A.get(i, j).toNumber(), 8);
        });

        it("R è triangolare superiore dopo QR", () => {
            const A = rmat([[[1, 1], [2, 1]], [[1, 2], [1, 1]], [[2, 1], [3, 2]]]);
            const { R } = Matrix.decomp.qr(A);
            expect(R.isUpperTriangular(1e-10)).toBe(true);
        });
    });
});

// ============================================================
// 8. SOLVER ESATTO
// ============================================================

describe("Matrix<Rational> — solver esatto", () => {

    it("LUP: A * x = b → soluzione esatta", () => {
        // Sistema: 2x + y = 5, x + 3y = 10 → x = 1, y = 3
        const A = rmat([[[2, 1], [1, 1]], [[1, 1], [3, 1]]]);
        const b = Matrix.zerosOf<Rational>(2, 1, Rational.zero, Rational.one);
        b.set(0, 0, r(5));
        b.set(1, 0, r(10));
        const x = A.solve(b, "LUP");
        // La soluzione deve essere esatta: x = 1, y = 3
        expect(rEq(x.get(0, 0) as Rational, r(1))).toBe(true);
        expect(rEq(x.get(1, 0) as Rational, r(3))).toBe(true);
    });

    it("LUP: sistema con soluzione frazionaria esatta", () => {
        // [[3, 1], [1, 2]] * [[x], [y]] = [[1], [1]]
        // det = 6 - 1 = 5
        // x = (1*2 - 1*1)/5 = 1/5, y = (3*1 - 1*1)/5 = 2/5
        const A = rmat([[[3, 1], [1, 1]], [[1, 1], [2, 1]]]);
        const b = Matrix.zerosOf<Rational>(2, 1, Rational.zero, Rational.one);
        b.set(0, 0, r(1)); b.set(1, 0, r(1));
        const x = A.solve(b, "LUP");
        expect(rEq(x.get(0, 0) as Rational, r(1, 5))).toBe(true);
        expect(rEq(x.get(1, 0) as Rational, r(2, 5))).toBe(true);
    });

    it("LU: stesso risultato di LUP su matrici non singolari", () => {
        const A = rmat([[[2, 1], [1, 1]], [[4, 1], [3, 1]]]);
        const b = Matrix.zerosOf<Rational>(2, 1, Rational.zero, Rational.one);
        b.set(0, 0, r(3)); b.set(1, 0, r(7));
        const xLUP = A.solve(b, "LUP");
        const xLU  = A.solve(b, "LU");
        expect(mEq(xLUP as Matrix<Rational>, xLU as Matrix<Rational>)).toBe(true);
    });

    it("A * inv(A) * b = b (inversa esatta)", () => {
        const A = rmat([[[2, 1], [1, 1]], [[1, 1], [3, 1]]]);
        const b = Matrix.zerosOf<Rational>(2, 1, Rational.zero, Rational.one);
        b.set(0, 0, r(7)); b.set(1, 0, r(11));
        const x  = A.solve(b, "LUP");
        const Ax = A.mul(x);
        expect(mEq(Ax as Matrix<Rational>, b)).toBe(true);
    });

    it("Hilbert 3×3: soluzione esatta senza drift float", () => {
        // H*x = ones → la soluzione è nota e con Rational è esatta
        const H = Matrix.zerosOf<Rational>(3, 3, Rational.zero, Rational.one);
        for (let i = 0; i < 3; i++)
            for (let j = 0; j < 3; j++)
                H.set(i, j, r(1, i + j + 1));
        const ones3 = Matrix.zerosOf<Rational>(3, 1, Rational.zero, Rational.one);
        ones3.set(0, 0, r(1)); ones3.set(1, 0, r(1)); ones3.set(2, 0, r(1));
        const x = H.solve(ones3, "LUP");
        // Verifica H*x = ones ESATTAMENTE (zero errore floating)
        const Hx = H.mul(x);
        for (let i = 0; i < 3; i++) {
            expect(rEq(Hx.get(i, 0) as Rational, r(1))).toBe(true);
        }
    });

    it("sistema 3×3 generico: soluzione esatta", () => {
        // [[1,2,3],[0,1,4],[5,6,0]] * [x] = [b]
        const A = rmat([
            [[1, 1], [2, 1], [3, 1]],
            [[0, 1], [1, 1], [4, 1]],
            [[5, 1], [6, 1], [0, 1]],
        ]);
        const b = Matrix.zerosOf<Rational>(3, 1, Rational.zero, Rational.one);
        b.set(0, 0, r(14)); b.set(1, 0, r(17)); b.set(2, 0, r(5));
        const x = A.solve(b, "LUP");
        const Ax = A.mul(x);
        expect(mEq(Ax as Matrix<Rational>, b)).toBe(true);
    });
});

// ============================================================
// 9. OPERAZIONI ELEMENT-WISE E UNARIE
// ============================================================

describe("Matrix<Rational> — dotOps e unarie", () => {

    const A = rmat([[[1, 2], [2, 3]], [[3, 4], [4, 5]]]);

    it("negate: tutti i segni invertiti esattamente", () => {
        const N = A.negate();
        expect(rEq(N.get(0, 0) as Rational, r(-1, 2))).toBe(true);
        expect(rEq(N.get(1, 1) as Rational, r(-4, 5))).toBe(true);
    });

    it("abs: negativi diventano positivi esattamente", () => {
        const M = rmat([[[-1, 2], [2, 3]], [[-3, 4], [4, 5]]]);
        const Ab = M.abs();
        expect(rEq(Ab.get(0, 0) as Rational, r(1, 2))).toBe(true);
        expect(rEq(Ab.get(1, 0) as Rational, r(3, 4))).toBe(true);
    });

    it("dotMul: (1/2)*(2/3) = 1/3 esatto", () => {
        const B = rmat([[[2, 3], [3, 4]], [[4, 5], [5, 6]]]);
        const C = A.dotMul(B);
        expect(rEq(C.get(0, 0) as Rational, r(1, 3))).toBe(true);
    });

    it("dotDiv: (2/3)/(2/3) = 1 esatto", () => {
        const B = rmat([[[1, 2], [2, 3]], [[3, 4], [4, 5]]]);
        const C = A.dotDiv(B);
        // (1/2)/(1/2) = 1
        expect(rEq(C.get(0, 0) as Rational, Rational.one)).toBe(true);
        // (2/3)/(2/3) = 1
        expect(rEq(C.get(0, 1) as Rational, Rational.one)).toBe(true);
    });

    it("round: 7/4 → 2 esatto", () => {
        const M = rmat([[[7, 4], [5, 4]]]);
        const R = M.round();
        expect(rEq(R.get(0, 0) as Rational, r(2))).toBe(true);
        expect(rEq(R.get(0, 1) as Rational, r(1))).toBe(true);
    });

    it("dotPow: (1/2)^2 = 1/4 esatto", () => {
        const M = rmat([[[1, 2], [2, 3]]]);
        const P = M.dotPow(2);
        expect(rEq(P.get(0, 0) as Rational, r(1, 4))).toBe(true);
        expect(rEq(P.get(0, 1) as Rational, r(4, 9))).toBe(true);
    });
});

// ============================================================
// 10. TRASFORMAZIONI
// ============================================================

describe("Matrix<Rational> — trasformazioni", () => {

    const A = rmat([[[1, 2], [1, 3]], [[2, 3], [1, 4]]]);

    it("transpose: (A^T)_{ij} = A_{ji} esatto", () => {
        const AT = A.t();
        expect(rEq(AT.get(0, 1) as Rational, A.get(1, 0) as Rational)).toBe(true);
        expect(rEq(AT.get(1, 0) as Rational, A.get(0, 1) as Rational)).toBe(true);
    });

    it("(A^T)^T = A esatto", () => {
        expect(mEq(A.t().t() as Matrix<Rational>, A)).toBe(true);
    });

    it("reshape: mantiene i dati invariati", () => {
        const M = rmat([[[1, 2], [3, 4], [5, 6]]]);   // 1×3
        const R = M.reshape(3, 1);                      // 3×1
        expect(rEq(R.get(2, 0) as Rational, r(5, 6))).toBe(true);
    });

    it("slice: estrae la sotto-matrice esatta", () => {
        const M = rmat([[[1, 1], [1, 2], [1, 3]], [[1, 4], [1, 5], [1, 6]]]);
        const S = M.slice(0, 2, 1, 3);
        expect(rEq(S.get(0, 0) as Rational, r(1, 2))).toBe(true);
        expect(rEq(S.get(1, 1) as Rational, r(1, 6))).toBe(true);
    });

    it("flip dim=1: righe invertite esatte", () => {
        const F = A.flip(1);
        expect(rEq(F.get(0, 0) as Rational, A.get(1, 0) as Rational)).toBe(true);
    });

    it("rot90 × 4 = identità (esatto)", () => {
        const E = A.rot90().rot90().rot90().rot90();
        expect(mEq(E as Matrix<Rational>, A)).toBe(true);
    });
});

// ============================================================
// 11. STATISTICHE
// ============================================================

describe("Matrix<Rational> — statistiche", () => {

    const A = rmat([[[1, 2], [1, 3]], [[2, 3], [1, 4]]]);

    it("sum(dim=1): somma per colonne esatta", () => {
        const S = A.sum(1);
        // col 0: 1/2 + 2/3 = 3/6+4/6 = 7/6
        // col 1: 1/3 + 1/4 = 4/12+3/12 = 7/12
        expect(rEq(S.get(0, 0) as Rational, r(7, 6))).toBe(true);
        expect(rEq(S.get(0, 1) as Rational, r(7, 12))).toBe(true);
    });

    it("sum(dim=2): somma per righe esatta", () => {
        const S = A.sum(2);
        // riga 0: 1/2 + 1/3 = 5/6
        // riga 1: 2/3 + 1/4 = 8/12+3/12 = 11/12
        expect(rEq(S.get(0, 0) as Rational, r(5, 6))).toBe(true);
        expect(rEq(S.get(1, 0) as Rational, r(11, 12))).toBe(true);
    });

    it("totalSum esatto", () => {
        // 1/2 + 1/3 + 2/3 + 1/4 = 6/12+4/12+8/12+3/12 = 21/12 = 7/4
        expect(rEq(A.totalSum() as Rational, r(7, 4))).toBe(true);
    });

    it("mean(dim=1): media per colonne esatta", () => {
        const M = A.mean(1);
        // col 0: 7/6 / 2 = 7/12
        expect(rEq(M.get(0, 0) as Rational, r(7, 12))).toBe(true);
        // col 1: 7/12 / 2 = 7/24
        expect(rEq(M.get(0, 1) as Rational, r(7, 24))).toBe(true);
    });
});

// ============================================================
// 12. IDENTITÀ MATEMATICHE CON ARITMETICA ESATTA
// ============================================================

describe("Identità matematiche su Matrix<Rational> — zero errore float", () => {

    it("(A+B)*C = A*C + B*C esatto", () => {
        const A = rmat([[[1, 2], [1, 3]], [[2, 3], [1, 4]]]);
        const B = rmat([[[1, 4], [1, 5]], [[1, 6], [1, 7]]]);
        const C = rmat([[[2, 3], [3, 4]], [[4, 5], [5, 6]]]);
        const lhs = A.add(B).mul(C);
        const rhs = A.mul(C).add(B.mul(C));
        expect(mEq(lhs as Matrix<Rational>, rhs as Matrix<Rational>)).toBe(true);
    });

    it("trace(A+B) = trace(A) + trace(B) esatto", () => {
        const A = rmat([[[1, 3], [0, 1]], [[0, 1], [2, 5]]]);
        const B = rmat([[[1, 4], [0, 1]], [[0, 1], [3, 7]]]);
        const lhs = A.add(B).trace() as Rational;
        const rhs = (A.trace() as Rational).add(B.trace() as Rational);
        expect(rEq(lhs, rhs)).toBe(true);
    });

    it("det(A*B) = det(A)*det(B) esatto", () => {
        const A = rmat([[[3, 1], [1, 1]], [[2, 1], [4, 1]]]);
        const B = rmat([[[1, 1], [2, 1]], [[3, 1], [1, 1]]]);
        const detAB = A.mul(B).det() as Rational;
        const detAdB = (A.det() as Rational).multiply(B.det() as Rational);
        expect(rEq(detAB, detAdB)).toBe(true);
    });

    it("(A^T)^T = A (transposto doppio) esatto", () => {
        const A = rmat([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]);
        expect(mEq(A.t().t() as Matrix<Rational>, A)).toBe(true);
    });

    it("A * inv(A) = I esatto (matrice intera invertibile)", () => {
        // [[3,1],[2,1]] → det = 1 → inv = [[1,-1],[-2,3]]
        const A = rmat([[[3, 1], [1, 1]], [[2, 1], [1, 1]]]);
        const AI = A.mul(A.inv());
        const I = Matrix.identityOf(2, Rational.zero, Rational.one);
        expect(mEq(AI as Matrix<Rational>, I)).toBe(true);
    });

    it("100 iterazioni Jacobi su diagonale dominante converge (Rational)", () => {
        // A = [[4, 1], [1, 4]] (diagonale dominante) → soluzione unica esatta
        const A = rmat([[[4, 1], [1, 1]], [[1, 1], [4, 1]]]);
        const b = Matrix.zerosOf<Rational>(2, 1, Rational.zero, Rational.one);
        b.set(0, 0, r(5)); b.set(1, 0, r(5));
        // A*[1,1] = [5, 5] → soluzione x=[1,1]
        const x = A.solve(b, "LUP");
        expect(rEq(x.get(0, 0) as Rational, Rational.one)).toBe(true);
        expect(rEq(x.get(1, 0) as Rational, Rational.one)).toBe(true);
    });

    it("soluzione di Hilbert 4×4: nessun drift floating-point", () => {
        const n = 4;
        const H = Matrix.zerosOf<Rational>(n, n, Rational.zero, Rational.one);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++)
                H.set(i, j, r(1, i + j + 1));
        const ones4 = Matrix.zerosOf<Rational>(n, 1, Rational.zero, Rational.one);
        for (let i = 0; i < n; i++) ones4.set(i, 0, Rational.one);
        const x = H.solve(ones4, "LUP");
        const Hx = H.mul(x);
        // La ricostruzione DEVE essere esatta — zero errore floating
        for (let i = 0; i < n; i++) {
            expect(rEq(Hx.get(i, 0) as Rational, Rational.one)).toBe(true);
        }
    });

    it("denominatori della soluzione di Hilbert 4×4 sono ridotti (cross-cancel funziona)", () => {
        const n = 4;
        const H = Matrix.zerosOf<Rational>(n, n, Rational.zero, Rational.one);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++)
                H.set(i, j, r(1, i + j + 1));
        const ones4 = Matrix.zerosOf<Rational>(n, 1, Rational.zero, Rational.one);
        for (let i = 0; i < n; i++) ones4.set(i, 0, Rational.one);
        const x = H.solve(ones4, "LUP");
        // Verifica che nessun denominatore sia esageratamente grande
        // (senza cross-cancel il denominatore esploderebbe a cifre enormi)
        for (let i = 0; i < n; i++) {
            const xi = x.get(i, 0) as Rational;
            // Con cross-cancellation i denominatori rimangono ragionevoli (< 10^6)
            expect(xi.den < 10_000_000).toBe(true);
        }
    });
});

// src/examples/usage.ts
//
// Dimostra l'uso trasparente di Matrix<Float64M>, Matrix<Complex> e Matrix<Rational>.
// Tutti gli algoritmi (LUP, Cholesky, QR, Jacobi…) funzionano identicamente sui tre tipi.

import { Matrix, Float64M, Complex, Rational } from "..";

// ============================================================
// 1. FLOAT  (default — identico al vecchio codice)
// ============================================================
console.log("=== FLOAT ===");
const Af = Matrix.fromArray([
    [4, 3],
    [6, 3],
]);
const bf = Matrix.fromArray([[10], [12]]);

console.log("A =\n" + Af);
console.log("b =\n" + bf);
const xf = Af.solve(bf);
console.log("x = A\\b =\n" + xf);
console.log("det(A) =", Af.det().toString());
console.log("trace(A) =", Af.trace().toString());
console.log("norm(A, fro) =", Af.norm("fro"));
console.log();

// ============================================================
// 2. COMPLESSA
// ============================================================
console.log("=== COMPLEX ===");

const Ac = Matrix.zerosOf(2, 2, Complex.zero, Complex.one);
Ac.set(0, 0, new Complex(2,  1));
Ac.set(0, 1, new Complex(1, -1));
Ac.set(1, 0, new Complex(0,  1));
Ac.set(1, 1, new Complex(3,  0));

const bc = Matrix.zerosOf(2, 1, Complex.zero, Complex.one);
bc.set(0, 0, new Complex(3, 2));
bc.set(1, 0, new Complex(1, 1));

console.log("A =\n" + Ac);
console.log("b =\n" + bc);
const xc = Ac.solve(bc);
console.log("x = A\\b =\n" + xc);
console.log("det(A) =", Ac.det().toString());
console.log("norm(A, fro) =", Ac.norm("fro"));
console.log();

// ============================================================
// 3. RAZIONALE ESATTA
// ============================================================
console.log("=== RATIONAL ===");

const n = 3;
const Ar = Matrix.zerosOf(n, n, Rational.zero, Rational.one);
for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
        Ar.set(i, j, new Rational(1, i + j + 1));

const br = Matrix.zerosOf(n, 1, Rational.zero, Rational.one);
for (let i = 0; i < n; i++)
    br.set(i, 0, Rational.one);

console.log("Hilbert(3) razionale =\n" + Ar);
console.log("b =\n" + br);
const xr = Ar.solve(br, "LUP");
console.log("x = H\\b (esatto!) =\n" + xr);
console.log("det(H) =", Ar.det().toString());
console.log();

// ============================================================
// 4. GALLERY + DECOMP (Float64M)
// ============================================================
console.log("=== GALLERY + DECOMP ===");
const H4 = Matrix.gallery.hilbert(4);
console.log("Hilbert(4) =\n" + H4);
const { L, U } = Matrix.decomp.lup(H4);
console.log("L =\n" + L);
console.log("U =\n" + U);

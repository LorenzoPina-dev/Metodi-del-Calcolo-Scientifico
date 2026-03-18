import { lup, solve } from '../src';
import { Matrix } from '../src/core/Matrix';

const EPS = 1e-10;

function assertApprox(a: number, b: number, msg?: string) {
  if (Math.abs(a - b) > EPS) throw new Error(msg ?? `Expected ${a} ≈ ${b}`);
}

function randomMatrix(n: number): Matrix {
  const m = new Matrix(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      m.set(i, j, Math.random() * 10);
    }
  }
  return m;
}

function benchmarkMatrix(n: number) {
  console.log(`\n--- TEST MATRICE ${n}x${n} ---`);
  const A = randomMatrix(n);

  // ---- LU decomposition ----
  console.time('LU');
  const { L, U, P } = lup(A);
  console.timeEnd('LU');

  // ---- Solve Ax = b ----
  const b = randomMatrix(n);
  console.time('Solve');
  const x = solve(A,b);
  console.timeEnd('Solve');

  // Verifica Ax ≈ b
  const Ax = A.multiply(x);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < b.cols; j++) {
      assertApprox(Ax.get(i, j), b.get(i, j), `Solve failed for size ${n}`);
    }
  }

  // ---- Inverse ----
  console.time('Inverse');
  const Ainv = A.inverse();
  console.timeEnd('Inverse');

  // Verifica A * Ainv ≈ I
  const I = A.multiply(Ainv);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      assertApprox(I.get(i, j), i === j ? 1 : 0, `Inverse failed for size ${n}`);
    }
  }

  // ---- Determinant ----
  console.time('Determinant');
  const det = A.det();
  console.timeEnd('Determinant');

  // ---- Transpose ----
  console.time('Transpose');
  const At = A.transpose();
  console.timeEnd('Transpose');

  // ---- Sum / Max ----
  console.time('Sum/Max');
  const totalSum = A.totalSum();
  const { value: maxVal } = A.max();
  console.timeEnd('Sum/Max');

  // ---- Slice / Add / Triangular ----
  console.time('Slice/Add/Triangular');
  if (n >= 2) {
    const slice = A.slice(0, 2, 0, 2);
    const rowAdd = new Matrix(1, n);
    for (let j = 0; j < n; j++) rowAdd.set(0, j, 1);
    const Bcast = A.add(rowAdd);
    const Utri = A.triu();
    const Ltri = A.tril();
  }
  console.timeEnd('Slice/Add/Triangular');

  console.log(`--- TEST MATRICE ${n}x${n} COMPLETED ---`);
}

// Genera matrici da 5x5 a 350x350
const sizes = [5, 10, 20, 50, 100, 150, 200, 250, 300, 350];
for (const n of sizes) {
  benchmarkMatrix(n);
}

console.log('--- ALL TESTS PASSED ---');
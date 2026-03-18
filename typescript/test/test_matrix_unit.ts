import { Matrix } from '../src/core/Matrix';

const EPS = 1e-10;

function assertApprox(a: number, b: number, msg?: string) {
  if (Math.abs(a - b) > EPS) throw new Error(msg ?? `Expected ${a} ≈ ${b}`);
}

function approxEqual(A: Matrix, B: Matrix, eps = EPS) {
  if (A.rows !== B.rows || A.cols !== B.cols) throw new Error('Shape mismatch');
  for (let i = 0; i < A.rows; i++)
    for (let j = 0; j < A.cols; j++)
      if (Math.abs(A.get(i, j) - B.get(i, j)) > eps) throw new Error(`Matrices differ at ${i},${j}`);
}

// 1) Clone behavior
(function testClone() {
  const A = new Matrix(2, 3);
  A.set(0, 0, 1.1);
  A.set(1, 2, 3.3);
  const C = A.clone();
  // values preserved
  approxEqual(A, C);
  // deep copy: modifying C does not change A
  C.set(0, 0, 9.9);
  if (Math.abs(A.get(0, 0) - 1.1) > EPS) throw new Error('Clone is not deep copy');
  console.log('testClone passed');
})();

// 2) Factories: identity, zeros, ones, diagFromArray
(function testFactories() {
  const I = Matrix.identity(3);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) assertApprox(I.get(i, j), i === j ? 1 : 0, 'Identity incorrect');

  const Z = Matrix.zeros(2, 4);
  assertApprox(Z.totalSum(), 0, 'Zeros incorrect');

  const O = Matrix.ones(2, 2);
  assertApprox(O.totalSum(), 4, 'Ones incorrect');

  const D = Matrix.diagFromArray([5, 6]);
  assertApprox(D.get(0, 0), 5);
  assertApprox(D.get(1, 1), 6);
  console.log('testFactories passed');
})();

// 3) Add broadcasting and incompatible shapes
(function testAddBroadcastingAndErrors() {
  const A = new Matrix(2, 3, new Float64Array([1, 2, 3, 4, 5, 6]));
  // row vector broadcast
  const row = new Matrix(1, 3, new Float64Array([1, 1, 1]));
  const sumRow = A.add(row);
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) assertApprox(sumRow.get(i, j), A.get(i, j) + 1);

  // column vector broadcast
  const col = new Matrix(2, 1, new Float64Array([10, 20]));
  const sumCol = A.add(col);
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) assertApprox(sumCol.get(i, j), A.get(i, j) + (i === 0 ? 10 : 20));

  // full-matrix add
  const B = Matrix.ones(2, 3);
  const C = A.add(B);
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) assertApprox(C.get(i, j), A.get(i, j) + 1);

  // incompatible shapes should throw
  let threw = false;
  try {
    A.add(Matrix.ones(3, 2));
  } catch (e) {
    threw = true;
  }
  if (!threw) throw new Error('add should throw on incompatible shapes');

  console.log('testAddBroadcastingAndErrors passed');
})();

// 4) Multiply and dimension mismatch
(function testMultiply() {
  const A = new Matrix(2, 3, new Float64Array([1, 2, 3, 4, 5, 6]));
  const B = new Matrix(3, 2, new Float64Array([7, 8, 9, 10, 11, 12]));
  const M = A.multiply(B);
  // manual multiplication check
  // M[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
  assertApprox(M.get(0, 0), 58);
  // M[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
  assertApprox(M.get(1, 1), 154);

  // mismatch should throw
  let threw = false;
  try {
    A.multiply(Matrix.ones(4, 4));
  } catch (e) {
    threw = true;
  }
  if (!threw) throw new Error('multiply should throw on dim mismatch');

  console.log('testMultiply passed');
})();

// 5) Sum, totalSum and max
(function testSumsAndMax() {
  const A = new Matrix(2, 3, new Float64Array([1, -2, 3, 4, 5, -6]));
  const colSum = A.sum(1); // axis=1 -> column vector (rows x 1)
  assertApprox(colSum.get(0, 0), 1 - 2 + 3);
  assertApprox(colSum.get(1, 0), 4 + 5 - 6);

  const rowSum = A.sum(0); // axis=0 -> row vector (1 x cols)
  assertApprox(rowSum.get(0, 0), 1 + 4);
  assertApprox(rowSum.get(0, 1), -2 + 5);
  assertApprox(rowSum.get(0, 2), 3 - 6);

  assertApprox(A.totalSum(), 1 - 2 + 3 + 4 + 5 - 6);

  const { value, row, col } = A.max();
  if (value !== 5 || row !== 1 || col !== 1) throw new Error('max returned wrong index/value');

  console.log('testSumsAndMax passed');
})();

// 6) Transpose
(function testTranspose() {
  const A = new Matrix(2, 3, new Float64Array([1, 2, 3, 4, 5, 6]));
  const T = A.transpose();
  if (T.rows !== 3 || T.cols !== 2) throw new Error('transpose shape incorrect');
  for (let i = 0; i < A.rows; i++) for (let j = 0; j < A.cols; j++) assertApprox(T.get(j, i), A.get(i, j));
  console.log('testTranspose passed');
})();

// 7) triu and tril
(function testTriuTril() {
  const A = new Matrix(3, 3, new Float64Array([1,2,3,4,5,6,7,8,9]));
  const U = A.triu();
  const L = A.tril();
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    if (j < i) assertApprox(U.get(i, j), 0);
    else assertApprox(U.get(i, j), A.get(i, j));
    if (j > i) assertApprox(L.get(i, j), 0);
    else assertApprox(L.get(i, j), A.get(i, j));
  }
  console.log('testTriuTril passed');
})();

// 8) isSymmetric
(function testIsSymmetric() {
  const S = new Matrix(2,2,new Float64Array([1,3,3,4]));
  if (!S.isSymmetric()) throw new Error('symmetric matrix reported non-symmetric');
  const NS = new Matrix(2,2,new Float64Array([1,2,3,4]));
  if (NS.isSymmetric()) throw new Error('non-symmetric matrix reported symmetric');
  console.log('testIsSymmetric passed');
})();

// 9) determinant and inverse consistency
(function testDetInverse() {
  const A = new Matrix(2,2,new Float64Array([4,7,2,6]));
  const detA = A.det();
  // det should be 4*6 - 7*2 = 10
  assertApprox(detA, 10);
  const Inv = A.inverse();
  const I = A.multiply(Inv);
  const Id = Matrix.identity(2);
  approxEqual(I, Id, 1e-6);
  console.log('testDetInverse passed');
})();

// 10) toString formatting
(function testToString() {
  const A = new Matrix(1,2,new Float64Array([1.2345, 6.789]));
  const s = A.toString();
  // should contain two numbers with 3 decimal places
  if (!/1.235/.test(s) || !/6.789/.test(s)) throw new Error('toString formatting incorrect');
  console.log('testToString passed');
})();

console.log('--- MATRIX UNIT TESTS COMPLETED ---');

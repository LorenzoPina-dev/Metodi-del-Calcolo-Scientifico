import { Matrix } from './src/matrici';

function approxEqual(A: Matrix, B: Matrix, eps = 1e-10) {
    if (A.rows !== B.rows || A.cols !== B.cols) return false;
    for (let i = 0; i < A.rows; i++)
        for (let j = 0; j < A.cols; j++)
            if (Math.abs(A.get(i,j)-B.get(i,j))>eps) return false;
    return true;
}

// -------- FACTORY TESTS --------
const I = Matrix.identity(3);
console.assert(I.get(0,0)===1 && I.get(1,1)===1 && I.get(2,2)===1, "Identity failed");
const Z = Matrix.zeros(2,2);
console.assert(Z.totalSum()===0, "Zeros failed");

// -------- ADD/SUBTRACT --------
const A = Matrix.diagFromArray([1,2,3]);
const B = Matrix.ones(3,3);
const C = A.add(B);
console.assert(C.get(0,0)===2 && C.get(1,1)===3, "Add failed");
const D = C.subtract(B);
console.assert(approxEqual(D,A), "Subtract failed");

// -------- MULTIPLY & TRANSPOSE --------
const E = A.multiply(B);
console.assert(E.get(0,0)===1+1+1, "Multiply failed");
const F = A.transpose();
console.assert(approxEqual(F,A), "Transpose failed");

// -------- LU / LUP --------
const M = new Matrix(3,3,new Float64Array([2,1,1,4,3,3,8,7,9]));
const {L,U,P} = M.lup();
const M_recon = L.multiply(U);
const M_perm = new Matrix(3,3);
for(let i=0;i<3;i++) for(let j=0;j<3;j++) M_perm.set(i,j,M.get(P[i],j));
console.assert(approxEqual(M_recon,M_perm), "LUP reconstruction failed");

// -------- SOLVE --------
const b = new Matrix(3,1,new Float64Array([1,2,3]));
const x = M.solve(b);
const b_recon = M.multiply(x);
console.assert(approxEqual(b,b_recon), "Solve failed");

// -------- DET / INVERSE --------
const det = M.det();
const M_inv = M.inverse();
const I_recon = M.multiply(M_inv);
console.assert(Math.abs(I_recon.get(0,0)-1)<1e-10, "Inverse failed");
console.assert(Math.abs(det-M.det())<1e-10, "Determinant failed");

// -------- CHOLESKY --------
const S = new Matrix(2,2,new Float64Array([4,2,2,3]));
const L_chol = Matrix.cholesky(S);
const S_recon = L_chol.multiply(L_chol.transpose());
console.assert(approxEqual(S,S_recon), "Cholesky failed");

// -------- FORWARD / BACKWARD SOLVE --------
const L_test = Matrix.identity(3).add(Matrix.ones(3,3)).tril();
const y = Matrix.solveLowerTriangular(L_test,b);
const b_check = L_test.multiply(y);
console.assert(approxEqual(b,b_check), "Forward substitution failed");

const U_test = Matrix.identity(3).add(Matrix.ones(3,3)).triu();
const x_back = Matrix.solveUpperTriangular(U_test,b);
const b_check2 = U_test.multiply(x_back);
console.assert(approxEqual(b,b_check2), "Backward substitution failed");

// -------- PERFORMANCE --------
const N = 200;
const big = Matrix.identity(N);
console.time("Multiply perf");
big.multiply(big);
console.timeEnd("Multiply perf");

console.time("Solve perf");
const b_big = Matrix.ones(N,1);
big.solve(b_big);
console.timeEnd("Solve perf");

console.log("All tests passed ✅");
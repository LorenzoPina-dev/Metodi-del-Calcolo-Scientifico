import {wathen} from "../src/init";
import { Matrix } from "../src"; 
// ===============================
// Utility
// ===============================

function isSymmetric(H: Matrix, tol = 1e-10): boolean {
    const n = H.rows;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (Math.abs(H.get(i, j) - H.get(j, i)) > tol) {
                console.log(`❌ Not symmetric at (${i}, ${j}) (${H.get(i, j)}, ${H.get(j, i)})`);
                return false;
            }
        }
    }
    return true;
}

function hasPositiveDiagonal(H: Matrix): boolean {
    for (let i = 0; i < H.rows; i++) {
        if (H.get(i, i) <= 0) {
            console.log(`❌ Non-positive diagonal at ${i}`);
            return false;
        }
    }
    return true;
}

function density(H: Matrix): number {
    let count = 0;
    const n = H.rows;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            if (H.get(i, j) !== 0) count++;
        }
    }
    return count / (n * n);
}


// ===============================
// Test principale
// ===============================

function testWathen(nx: number, ny: number) {
    console.log(`\n=== TEST nx=${nx}, ny=${ny} ===`);

    const H = wathen(nx, ny);

    const expectedSize =  3 * nx * ny + 2 * nx + 2 * ny + 1;

    // 1. Dimensione
    if (H.rows !== expectedSize || H.cols !== expectedSize) {
        console.log(`❌ Dimension error: got ${H.rows}, expected ${expectedSize}`);
        return;
    } else {
        console.log("✅ Dimension OK");
    }

    // 2. Simmetria
    console.log(isSymmetric(H) ? "✅ Symmetric" : "❌ Symmetry failed");

    // 3. Diagonale positiva
    console.log(hasPositiveDiagonal(H) ? "✅ Positive diagonal" : "❌ Diagonal error");

    // 4. Densità
    const d = density(H);
    console.log(`ℹ️ Density: ${d}`);
    if (d > 0.1) {
        console.log("⚠️ Too dense (possible assembly issue)");
    } else {
        console.log("✅ Sparsity OK");
    }

    // 5. Definita positiva
    console.log(Matrix.decomp.cholesky(H) ? "✅ Positive definite" : "❌ Not positive definite");
}

// ===============================
// Test multipli
// ===============================

function runTests() {
    const cases = [
        [1, 1],
        [2, 2],
        [3, 3],
        [5, 5],
        [10, 10]
    ];

    for (const [nx, ny] of cases) {
        testWathen(nx, ny);
    }
}

runTests();
import fs from "fs";
import { Matrix } from "../src/core/Matrix"; // Assicurati che il path punti alla tua classe Matrix

// Configurazione
const minSize = 5;
const maxSize = 500;
const step = 50;

const results: Record<number, any> = {};

const testSizes: number[] = [];
for (let n = minSize; n <= maxSize; n += step) testSizes.push(n);

console.log("Benchmark in corso...");

for (const n of testSizes) {
    console.log(`Matrix size: ${n}x${n}`);
    // Matrici di test
    const A = Matrix.ones(n, n).add(Matrix.diag(n, n)); // per evitare singolarità
    
    const b = Matrix.ones(n, 1);

    const timing: Record<string, number> = {};

    // --- LUP Decomposition ---
    let start = Date.now();
    const { L, U, P } = A.lup();
    timing["LUP"] = Date.now() - start;

    // --- LU Decomposition (no pivot) ---
    start = Date.now();
    Matrix.lu(A);
    timing["LU"] = Date.now() - start;

    // --- LU Total Pivoting ---
    start = Date.now();
    Matrix.luPivotingTotal(A);
    timing["LU_TotalPivot"] = Date.now() - start;

    // --- Solve system ---
    start = Date.now();
    A.solve(b);
    timing["Solve"] = Date.now() - start;

    // --- Determinant ---
    start = Date.now();
    A.det();
    timing["Determinant"] = Date.now() - start;

    // --- Inverse ---
    start = Date.now();
    A.inverse();
    timing["Inverse"] = Date.now() - start;

    results[n] = timing;
}

// Salva tutto in JSON
fs.writeFileSync("benchmark_results.json", JSON.stringify(results, null, 2));
console.log("Benchmark completato. Risultati salvati in benchmark_results.json");
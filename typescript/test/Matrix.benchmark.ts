// Matrix.fullBenchmark.ts
import { writeFileSync } from 'fs';
import { Matrix } from '../src';

// dimensioni da testare
const sizes = [50, 100, 150, 200, 250, 300];
const iterations = 3; // numero di ripetizioni per media

// misura il tempo medio di una funzione
function measure(fn: () => any, reps = iterations): number {
    const times: number[] = [];
    for (let i = 0; i < reps; i++) {
        const start = performance.now();
        fn();
        const end = performance.now();
        times.push(end - start);
    }
    return times.reduce((a,b)=>a+b,0)/reps;
}

// lista dei metodi principali da testare
const benchmarkMethods: { name: string, fn: (A: Matrix, B?: Matrix) => any }[] = [
    // ARITMETICA
    { name: 'add', fn: (A,B) => A.add(B!) },
    { name: 'sub', fn: (A,B) => A.sub(B!) },
    { name: 'mul', fn: (A,B) => A.mul(B!) },
    { name: 'pow', fn: (A) => A.pow(2) },

    // DOT OPS
    { name: 'dotMul', fn: (A,B) => A.dotMul(B!) },
    { name: 'dotDiv', fn: (A,B) => A.dotDiv(B!) },
    { name: 'dotPow', fn: (A) => A.dotPow(2) },

    // ALGEBRA
    { name: 'trace', fn: (A) => A.trace() },
    { name: 'totalSum', fn: (A) => A.totalSum() },
    { name: 'norm', fn: (A) => A.norm() },
    { name: 'inv', fn: (A) => A.isSquare() ? A.inv() : null },
    { name: 'det', fn: (A) => A.isSquare() ? A.det() : null },

    // STATISTICHE
    { name: 'sum', fn: (A) => A.sum(1) },
    { name: 'mean', fn: (A) => A.mean(1) },
    { name: 'max', fn: (A) => A.max() },
    { name: 'min', fn: (A) => A.min() },

    // TRASFORMAZIONI
    { name: 't', fn: (A) => A.t() },
    { name: 'reshape', fn: (A) => A.reshape(Math.floor(A.rows/2), A.cols*2) },
    { name: 'repmat', fn: (A) => A.repmat(2,2) },
    { name: 'slice', fn: (A) => A.slice(0, Math.floor(A.rows/2), 0, Math.floor(A.cols/2)) },
    { name: 'flip', fn: (A) => A.flip(1) },
    { name: 'rot90', fn: (A) => A.rot90() },

    // UNARY
    { name: 'abs', fn: (A) => A.abs() },
    { name: 'sqrt', fn: (A) => A.sqrt() },
    { name: 'round', fn: (A) => A.round() },

    // PROPRIETA'
    { name: 'isSquare', fn: (A) => A.isSquare() },
    { name: 'isSymmetric', fn: (A) => A.isSymmetric() },
    { name: 'isUpperTriangular', fn: (A) => A.isUpperTriangular() },
    { name: 'isLowerTriangular', fn: (A) => A.isLowerTriangular() },
    { name: 'isDiagonal', fn: (A) => A.isDiagonal() },
    { name: 'isIdentity', fn: (A) => A.isIdentity() },
    { name: 'isOrthogonal', fn: (A) => A.isOrthogonal() },
    { name: 'isZeroMatrix', fn: (A) => A.isZeroMatrix() },
];

// struttura per salvare i risultati
const results: Record<number, Record<string, number>> = {};

(async () => {
    for (const N of sizes) {
        console.log(`\nBenchmarking size: ${N}x${N}`);
        const A = Matrix.random(N, N);
        const B = Matrix.random(N, N);
        results[N] = {};

        for (const method of benchmarkMethods) {
            try {
                const time = measure(() => method.fn(A, B));
                results[N][method.name] = parseFloat(time.toFixed(3));
                console.log(`${method.name}: ${time.toFixed(3)} ms`);
            } catch(e) {
                results[N][method.name] = -1; // errore o non applicabile
                console.log(`${method.name}: skipped`);
            }
        }
    }

    // salva i risultati su file JSON
    writeFileSync('matrix_full_benchmark_results.json', JSON.stringify(results, null, 2));
    console.log('\nBenchmark completo. Risultati salvati in matrix_full_benchmark_results.json');
})();
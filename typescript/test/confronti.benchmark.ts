import { performance } from 'perf_hooks';
import { Matrix } from '../src';

const sizes = [10, 100, 200, 400];

interface MegaBenchResult {
    Dim: number;
    Tipo_Matrice: string;
    Metodo: string;
    Approccio: 'Statico' | 'Dinamico';
    Tempo_ms: number;
    Memoria_MB: number;
    Residuo_Rel: string;
    Stato: string;
}

async function runUltimateBenchmark() {
    console.log("🚀 AVVIO SUPREMO BENCHMARK: Analisi Comparativa Totale\n");
    const results: MegaBenchResult[] = [];

    for (const N of sizes) {
        const x_true = Matrix.random(N, 1);

        // --- PREPARAZIONE SCENARI ---
        const scenarios = [
            { 
                name: 'SPD', 
                A: makeSPD(N),
                solvers: ['CHOLESKY', 'LU', 'LUP', 'QR', 'JACOBI', 'GAUSS-SEIDEL'] 
            },
            {
                name: 'DiagonalDominant',
                A: makeDiagonallyDominant(N),
                solvers: ['JACOBI', 'GAUSS-SEIDEL', 'LUP']
            },
            {
                name: 'General',
                A: makeGeneralWellConditioned(N),
                solvers: ['LU', 'LUP', 'QR']
            },
            {
                name: 'Wilkinson',
                A: Matrix.gallery.wilkinson(N),
                solvers: ['LUP', 'QR'] // LU escluso ❗
            },
            {
                name: 'Hanowa',
                A: (N % 2 === 0) ? Matrix.gallery.hanowa(N) : Matrix.gallery.hanowa(N + 1),
                solvers: ['LUP', 'QR']
            },
            {
                name: 'Dorr',
                A: Matrix.gallery.dorr(N),
                solvers: ['LUP', 'JACOBI'] // GS può divergere
            }
        ];

        for (const scenario of scenarios) {
            const b = scenario.A.mul(x_true);
            const currentN = scenario.A.rows;

            for (const sName of scenario.solvers) {
                // Testiamo sia la versione Statica (In-place) che Dinamica
                // Nota: se il tuo metodo non ha la variante, lo script cattura l'errore
                ['Dinamico', 'Statico'].forEach(approccio => {
                    results.push(executeTest(currentN, scenario.name, sName, approccio as any, scenario.A, b, x_true));
                });
            }
        }
    }

    console.table(results);
}

function executeTest(N: number, type: string, method: string, approach: 'Statico' | 'Dinamico', A: Matrix, b: Matrix, x_true: Matrix): MegaBenchResult {
    if (global.gc) global.gc();
    const memStart = process.memoryUsage().heapUsed;
    
    let errorRel = 0;
    let status = "✅ OK";
   /* console.log(`\n🔍 La matrice è: `);
    if(A.isSymmetric()) console.log(`\n A è simmetrica`);
    if(A.isUpperTriangular()) console.log(`\n A è triangolare superiore`);
    if(A.isLowerTriangular()) console.log(`\n A è triangolare inferiore`);
    if(A.isInvertible()) console.log(`\n A è invertibile`);
    if(A.isOrthogonal()) console.log(`\n A è ortogonale`);
    if(A.isDiagonallyDominant()) console.log(`\n A è matrice diagonale dominante`);*/



    try {
        const start = performance.now();
        let x_calc= A.solve(b, method);

        const end = performance.now();
        const memEnd = process.memoryUsage().heapUsed;

        // Calcolo Residuo Relativo
        const res = A.mul(x_calc).sub(b);
        errorRel = normInf(res) / (normInfMat(A) * normInf(x_calc) + normInf(b));
        
        if (errorRel > 1e-5) status = "⚠️ Instabile";
        if (isNaN(errorRel)) status = "❌ Fallito";
        if (!isMethodApplicable(A, method)) {
            return {
                Dim: N,
                Tipo_Matrice: type,
                Metodo: method,
                Approccio: approach,
                Tempo_ms: 0,
                Memoria_MB: 0,
                Residuo_Rel: "N/A",
                Stato: "🚫 Non Applicabile"
            };
        }
        return {
            Dim: N,
            Tipo_Matrice: type,
            Metodo: method,
            Approccio: approach,
            Tempo_ms: Number((end - start).toFixed(3)),
            Memoria_MB: Number(((memEnd - memStart) / 1024 / 1024).toFixed(3)),
            Residuo_Rel: errorRel.toExponential(2),
            Stato: status
        };
    } catch (e: any) {
        console.error(`Errore in ${method}  per N=${N}:`, e.message);
        return {
            Dim: N, Tipo_Matrice: type, Metodo: method, Approccio: approach,
            Tempo_ms: 0, Memoria_MB: 0, Residuo_Rel: "N/A", Stato: "🚫 Non Imp."
        };
    }
}


// Helper Norme (0-based)
function normInf(v: Matrix) {
    let max = 0;
    for (let i = 0; i < v.rows; i++) max = Math.max(max, Math.abs(v.get(i, 0)));
    return max;
}
function normInfMat(m: Matrix) {
    let max = 0;
    for (let i = 0; i < m.rows; i++) {
        let sum = 0;
        for (let j = 0; j < m.cols; j++) sum += Math.abs(m.get(i, j));
        max = Math.max(max, sum);
    }
    return max;
}
function isMethodApplicable(A: Matrix, method: string): boolean {
    switch (method.toUpperCase()) {
        case 'CHOLESKY':
            return A.isSymmetric() && A.isPositiveDefinite();

        case 'LU':
            return A.isSquare();

        case 'LUP':
            return A.isSquare();

        case 'QR':
            return true;

        case 'JACOBI':
        case 'GAUSS-SEIDEL':
            return A.isDiagonallyDominant() || A.isPositiveDefinite();

        default:
            return true;
    }
}

runUltimateBenchmark();

function makeSPD(N: number): Matrix {
    const M = Matrix.random(N, N);
    const A = M.t().mul(M);
    return A.add(Matrix.identity(N).mul(N * 1e-3));
}

function makeDiagonallyDominant(N: number): Matrix {
    const A = Matrix.random(N, N);

    for (let i = 0; i < N; i++) {
        let rowSum = 0;
        for (let j = 0; j < N; j++) {
            if (i !== j) rowSum += Math.abs(A.get(i, j));
        }
        A.set(i, i, rowSum + 1); // forza dominanza
    }

    return A;
}

function makeGeneralWellConditioned(N: number): Matrix {
    return Matrix.random(N, N).add(Matrix.identity(N).mul(0.5));
}
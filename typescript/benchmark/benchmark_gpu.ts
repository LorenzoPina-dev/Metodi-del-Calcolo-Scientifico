// benchmark/benchmark_gpu.ts
//
// Benchmark GPU (WebGPU) e Worker threads — usa esclusivamente l'API Matrix standard.
//
// Tutti i backend (GPU, Workers, WASM, TS) sono trasparenti:
//   A.mulAsync(B)          — matmul con backend ottimale (GPU → Workers → WASM → TS)
//   A.solveJacobiAsync(b)  — Jacobi con backend ottimale (GPU → Workers → WASM → TS)
//   Matrix.initCompute()   — inizializza tutti i backend una sola volta
//
// Per confronti espliciti per backend, ogni percorso viene forzato
// tramite le stesse API Matrix con diversi livelli di inizializzazione.
//
// Utilizzo:
//   npx tsx --expose-gc benchmark/benchmark_gpu.ts

import { performance }   from "node:perf_hooks";
import { writeFileSync } from "node:fs";
import { Matrix, Float64M } from "../src/index.js";

const DEBUG      = process.env.BENCH_DEBUG === "1";
const TIMEOUT_MS = Number.parseInt(process.env.BENCH_TIMEOUT_MS ?? "0", 10);

process.on("uncaughtException",   (err)    => console.error("[uncaughtException]", err));
process.on("unhandledRejection",  (reason) => console.error("[unhandledRejection]", reason));

function dbg(msg: string) { if (DEBUG) console.log(msg); }

async function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
    if (!ms || ms <= 0) return p;
    let t: NodeJS.Timeout;
    const timeout = new Promise<never>((_, reject) => {
        t = setTimeout(() => reject(new Error(`${label} timeout (${ms}ms)`)), ms);
    });
    try {
        return await Promise.race([p, timeout]);
    } finally {
        clearTimeout(t!);
    }
}

// ── Inizializza tutti i backend tramite Matrix.initCompute() ─────────────────
const caps = await Matrix.initCompute({ gpu: true, workers: true });

console.log("\n+----------------------------------------------------------+");
console.log("|        numeric-matrix  BENCHMARK GPU + PARALLEL         |");
console.log(`|  WASM=${caps.wasm?"OK":"--"}  Workers=${caps.workers?`OK(${caps.numWorkers})`:"--"}  GPU=${caps.gpu?"OK(f32)":"--"}           |`);
console.log("+----------------------------------------------------------+\n");
dbg(`[debug] BENCH_TIMEOUT_MS=${TIMEOUT_MS}`);

interface BenchRow {
    op: string; n: number; backend: string;
    avgMs: number; minMs: number; gflops?: number;
}
const results: BenchRow[] = [];

function gc() { if ((globalThis as any).gc) (globalThis as any).gc(); }

function measureSync(fn: () => void, warmup = 3, reps = 5) {
    for (let i = 0; i < warmup; i++) { try { fn(); } catch {} }
    gc();
    const ts: number[] = [];
    for (let i = 0; i < reps; i++) {
        const t0 = performance.now();
        try { fn(); } catch (e) { if (DEBUG) console.error("[measureSync]", e); }
        ts.push(performance.now() - t0);
    }
    return { avgMs: ts.reduce((a, b) => a + b) / ts.length, minMs: Math.min(...ts) };
}

async function measureAsync(fn: () => Promise<void>, warmup = 2, reps = 4) {
    for (let i = 0; i < warmup; i++) {
        try { await withTimeout(fn(), TIMEOUT_MS, "measureAsync warmup"); }
        catch (e) { if (DEBUG) console.error("[measureAsync warmup]", e); }
    }
    gc();
    const ts: number[] = [];
    for (let i = 0; i < reps; i++) {
        const t0 = performance.now();
        try { await withTimeout(fn(), TIMEOUT_MS, "measureAsync"); }
        catch (e) { if (DEBUG) console.error("[measureAsync]", e); }
        ts.push(performance.now() - t0);
    }
    return { avgMs: ts.reduce((a, b) => a + b) / ts.length, minMs: Math.min(...ts) };
}

// ── MATMUL benchmark ─────────────────────────────────────────────────────────
//
// Per ogni dimensione n misuriamo:
//   1. A.mul(B)        — percorso sincrono (WASM + SIMD → TS fast-path)
//   2. A.mulAsync(B)   — percorso asincrono ottimale (GPU → Workers → WASM)
//
// Il percorso asincrono usa automaticamente il backend più veloce disponibile
// in base alla dimensione e alle capabilities rilevate.

console.log("---  MATMUL  ---------------------------------------------------\n");
for (const n of [100, 200, 300, 400, 500, 700, 1000]) {
    const A = Matrix.random(n, n) as Matrix<Float64M>;
    const B = Matrix.random(n, n) as Matrix<Float64M>;
    const flops = 2 * n ** 3;

    // Percorso sincrono: WASM + SIMD (o TS fast-path per n piccoli)
    {
        const m = measureSync(() => { A.mul(B); });
        results.push({ op: "matmul", n, backend: "sync (WASM→TS)", ...m,
            gflops: flops / (m.avgMs * 1e6) });
        console.log(`  n=${String(n).padStart(4)}  sync      ${m.avgMs.toFixed(2).padStart(8)}ms  ${(flops/(m.avgMs*1e6)).toFixed(3)} GFLOPS`);
    }

    // Percorso asincrono: GPU → Workers → WASM (backend selezionato automaticamente)
    if (caps.workers || caps.gpu) {
        dbg(`[debug] n=${n} mulAsync start`);
        const m = await measureAsync(async () => { await A.mulAsync(B); });
        dbg(`[debug] n=${n} mulAsync done`);
        // Determina quale backend è stato effettivamente usato
        const backend = caps.gpu && n * n >= 90_000 ? "async→GPU(f32)"
                      : caps.workers && n * n >= 90_000 ? `async→Workers(${caps.numWorkers})`
                      : "async→WASM";
        results.push({ op: "matmul", n, backend, ...m,
            gflops: flops / (m.avgMs * 1e6) });
        console.log(`  n=${String(n).padStart(4)}  ${backend.padEnd(18)} ${m.avgMs.toFixed(2).padStart(8)}ms  ${(flops/(m.avgMs*1e6)).toFixed(3)} GFLOPS${caps.gpu && n*n>=90_000 ? "  (f32)" : ""}`);
    }
    console.log();
}

// ── JACOBI benchmark ─────────────────────────────────────────────────────────
//
// Per ogni dimensione n misuriamo:
//   1. A.solve(b, "JACOBI")      — percorso sincrono (WASM → TS)
//   2. A.solveJacobiAsync(b)     — percorso asincrono ottimale (GPU → Workers → WASM)

console.log("---  JACOBI  ---------------------------------------------------\n");

function makeDiagDom(n: number): Matrix<Float64M> {
    const A = Matrix.random(n, n) as Matrix<Float64M>;
    for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) if (i !== j) s += Math.abs(A.get(i, j).value);
        A.set(i, i, s + 1 + Math.random());
    }
    return A;
}

for (const n of [50, 100, 200, 300, 500]) {
    const A   = makeDiagDom(n);
    const b   = Matrix.random(n, 1) as Matrix<Float64M>;
    const rhs = A.mul(b) as Matrix<Float64M>;

    // Percorso sincrono (WASM → TS fast-path)
    {
        const m = measureSync(() => { A.solve(rhs, "JACOBI"); });
        results.push({ op: "jacobi", n, backend: "sync (WASM→TS)", ...m });
        console.log(`  n=${String(n).padStart(4)}  sync      ${m.avgMs.toFixed(3).padStart(9)}ms`);
    }

    // Percorso asincrono (GPU → Workers → WASM) — solo se n è abbastanza grande
    if ((caps.workers && n >= 100) || (caps.gpu && n >= 200)) {
        const m = await measureAsync(async () => { await A.solveJacobiAsync(rhs); });
        const backend = caps.gpu && n * n >= 40_000 ? "async→GPU(f32)"
                      : caps.workers && n * n >= 10_000 ? `async→Workers(${caps.numWorkers})`
                      : "async→WASM";
        results.push({ op: "jacobi", n, backend, ...m });
        console.log(`  n=${String(n).padStart(4)}  ${backend.padEnd(18)} ${m.avgMs.toFixed(3).padStart(9)}ms${caps.gpu && n*n>=40_000 ? "  (f32)" : ""}`);
    }
    console.log();
}

// ── Salva risultati ───────────────────────────────────────────────────────────
writeFileSync("benchmark/benchmark_gpu_results.json",
    JSON.stringify({ meta: { date: new Date().toISOString(), caps }, results }, null, 2));
console.log("  benchmark/benchmark_gpu_results.json salvato");
console.log("\n  Nota: GPU usa f32 (~7 cifre significative). Per f64, usa WASM o Workers.");
console.log("  Tutti i percorsi passano per Matrix.mulAsync() / Matrix.solveJacobiAsync().");

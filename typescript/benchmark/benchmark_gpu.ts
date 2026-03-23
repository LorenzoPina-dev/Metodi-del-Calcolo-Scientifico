// benchmark/benchmark_gpu.ts
//
// Benchmark specifico per GPU (WebGPU) e Worker threads.
// Confronta le prestazioni di matmul e Jacobi tra:
//   • TS fast-path
//   • WASM + SIMD
//   • Worker threads (CPU parallelo)
//   • WebGPU (GPU, f32)
//
// Utilizzo:
//   npx tsx --expose-gc benchmark/benchmark_gpu.ts

import { performance }   from "node:perf_hooks";
import { writeFileSync } from "node:fs";
import { Matrix, Float64M } from "../src/index.js";
import { initCompute, getCapabilities } from "../src/compute.js";
import { mulAsync }           from "../src/ops/multiply.js";
import { solveJacobiAsync }   from "../src/solver/jacobi.js";
import { getBridgeSync }      from "../src/wasm/wasm_bridge.js";
import { getPoolSync }        from "../src/parallel/worker_pool.js";
import { isGPUAvailable }     from "../src/gpu/webgpu_backend.js";

// ── Inizializza tutti i backend ───────────────────────────────────────────────
const caps = await initCompute({ gpu: true, workers: true });
console.log("\n╔══════════════════════════════════════════════════════════╗");
console.log("║        numeric-matrix — BENCHMARK GPU + PARALLEL        ║");
console.log(`║  WASM=${caps.wasm?"✓":"✗"}  Workers=${caps.workers?`✓(${caps.numWorkers})`:"✗"}  GPU=${caps.gpu?"✓(f32)":"✗"}         ║`);
console.log("╚══════════════════════════════════════════════════════════╝\n");

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
        const t0 = performance.now(); try { fn(); } catch {}
        ts.push(performance.now() - t0);
    }
    return { avgMs: ts.reduce((a,b)=>a+b)/ts.length, minMs: Math.min(...ts) };
}

async function measureAsync(fn: () => Promise<void>, warmup = 2, reps = 4) {
    for (let i = 0; i < warmup; i++) { try { await fn(); } catch {} }
    gc();
    const ts: number[] = [];
    for (let i = 0; i < reps; i++) {
        const t0 = performance.now(); try { await fn(); } catch {}
        ts.push(performance.now() - t0);
    }
    return { avgMs: ts.reduce((a,b)=>a+b)/ts.length, minMs: Math.min(...ts) };
}

// ── MATMUL benchmark ──────────────────────────────────────────────────────────
console.log("━━━  MATMUL  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
for (const n of [100, 200, 300, 400, 500, 700, 1000]) {
    const A = Matrix.random(n, n);
    const B = Matrix.random(n, n);
    const flops = 2 * n ** 3;

    // TS fast-path
    {
        const m = measureSync(() => {
            // Forzato: bypassa WASM/GPU/Workers usando solo TS
            const af = A.data as any, bf = B.data as any;
            const cf = new Float64Array(n * n);
            for (let i=0;i<n;i++) { const iK=i*n,iN=i*n; for (let k=0;k<n;k++) { const aik=(af[iK+k]).value; if(aik===0)continue; const kN=k*n; for(let j=0;j<n;j++) cf[iN+j]+=aik*(bf[kN+j]).value; } }
        });
        results.push({ op:"matmul", n, backend:"TS fast-path", ...m, gflops: flops/(m.avgMs*1e6) });
        console.log(`  n=${String(n).padStart(4)}  TS        ${m.avgMs.toFixed(2).padStart(8)}ms  ${(flops/(m.avgMs*1e6)).toFixed(3)} GFLOPS`);
    }

    // WASM
    {
        const w = getBridgeSync();
        if (w) {
            const m = measureSync(() => { A.mul(B); });
            results.push({ op:"matmul", n, backend:"WASM+SIMD", ...m, gflops: flops/(m.avgMs*1e6) });
            console.log(`  n=${String(n).padStart(4)}  WASM+SIMD ${m.avgMs.toFixed(2).padStart(8)}ms  ${(flops/(m.avgMs*1e6)).toFixed(3)} GFLOPS`);
        }
    }

    // Workers
    if (caps.workers && n >= 100) {
        const m = await measureAsync(async () => { await mulAsync(A, B); });
        results.push({ op:"matmul", n, backend:`Workers(${caps.numWorkers})`, ...m, gflops: flops/(m.avgMs*1e6) });
        console.log(`  n=${String(n).padStart(4)}  Workers   ${m.avgMs.toFixed(2).padStart(8)}ms  ${(flops/(m.avgMs*1e6)).toFixed(3)} GFLOPS`);
    }

    // GPU
    if (caps.gpu && n >= 300) {
        try {
            const m = await measureAsync(async () => { await mulAsync(A, B); });
            results.push({ op:"matmul", n, backend:"GPU(f32)", ...m, gflops: flops/(m.avgMs*1e6) });
            console.log(`  n=${String(n).padStart(4)}  GPU(f32)  ${m.avgMs.toFixed(2).padStart(8)}ms  ${(flops/(m.avgMs*1e6)).toFixed(3)} GFLOPS  ⚠ f32`);
        } catch (e) {
            console.log(`  n=${String(n).padStart(4)}  GPU: ${(e as Error).message}`);
        }
    }
    console.log();
}

// ── JACOBI benchmark ──────────────────────────────────────────────────────────
console.log("━━━  JACOBI  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

function makeDiagDom(n: number): Matrix<Float64M> {
    const A = Matrix.random(n, n);
    for (let i=0;i<n;i++) {
        let s=0; for(let j=0;j<n;j++) if(i!==j) s+=Math.abs(A.get(i,j).value);
        A.set(i,i,s+1+Math.random());
    }
    return A;
}

for (const n of [50, 100, 200, 300, 500]) {
    const A = makeDiagDom(n);
    const b = Matrix.random(n, 1);
    const rhs = A.mul(b);

    // WASM Jacobi
    {
        const m = measureSync(() => { A.solve(rhs, "JACOBI"); });
        results.push({ op:"jacobi", n, backend:"WASM+SIMD", ...m });
        console.log(`  n=${String(n).padStart(4)}  WASM+SIMD ${m.avgMs.toFixed(3).padStart(9)}ms`);
    }

    // Workers
    if (caps.workers && n >= 100) {
        const m = await measureAsync(async () => { await solveJacobiAsync(A, rhs); });
        results.push({ op:"jacobi", n, backend:`Workers(${caps.numWorkers})`, ...m });
        console.log(`  n=${String(n).padStart(4)}  Workers   ${m.avgMs.toFixed(3).padStart(9)}ms`);
    }

    // GPU
    if (caps.gpu && n >= 200) {
        try {
            const m = await measureAsync(async () => { await solveJacobiAsync(A, rhs); });
            results.push({ op:"jacobi", n, backend:"GPU(f32)", ...m });
            console.log(`  n=${String(n).padStart(4)}  GPU(f32)  ${m.avgMs.toFixed(3).padStart(9)}ms  ⚠ f32`);
        } catch {}
    }
    console.log();
}

// ── Salva risultati ───────────────────────────────────────────────────────────
writeFileSync("benchmark/benchmark_gpu_results.json",
    JSON.stringify({ meta: { date: new Date().toISOString(), caps }, results }, null, 2));
console.log("  ✓ benchmark/benchmark_gpu_results.json");
console.log("\n  Nota: GPU usa f32 (~7 cifre significative). Per f64, usa WASM o Workers.");

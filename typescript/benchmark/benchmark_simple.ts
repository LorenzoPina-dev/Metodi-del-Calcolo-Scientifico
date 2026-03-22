// benchmark/benchmark_simple.ts
//
// Benchmark operazioni ELEMENTARI (O(n) e O(n²)) su matrici grandi.
//
// ── PERCHÉ LE DIMENSIONI SONO LIMITATE ────────────────────────────────────
// Ogni elemento di Matrix<Float64M> è un oggetto JS separato (Float64M).
// V8 alloca ~48 byte per oggetto (header + value + kind ref + alignment).
//
//   n=1000: 1M oggetti × 48B = 48MB per matrice
//   n=2000: 4M oggetti × 48B = 192MB per matrice
//   n=3000: 9M oggetti × 48B = 432MB per matrice  ← OOM con 3 matrici in heap
//
// Per operazioni binarie (add, dotMul) che tengono A + B + output in memoria:
//   n=2000: 3 × 192MB = 576MB → limite sicuro (~4GB heap Node di default)
//   n=3000: 3 × 432MB = 1.3GB → ok ma lento per GC
//   n=5000: 3 × 1.2GB = 3.6GB → OOM quasi certo
//
// SOLUZIONE ARCHITETTURALE:
//   Per grandi n senza overhead-oggetti, usare Float64Array direttamente.
//   Il benchmark mostra le PRESTAZIONI REALI della libreria con l'astrazione
//   generica — utile per capire dove agire sull'allocator.
//
// Per forzare più heap: node --max-old-space-size=8192 (8GB)
//
// Utilizzo: npx tsx --expose-gc benchmark/benchmark_simple.ts
//           npx tsx --expose-gc --max-old-space-size=8192 benchmark/benchmark_simple.ts
//

import { writeFileSync } from "node:fs";
import { performance }   from "node:perf_hooks";
import { Matrix, Float64M } from "../src/index.js";

// ── CONFIG ──────────────────────────────────────────────────────────────────
// Dimensioni scelte per non OOM con operazioni binarie (3 matrici in heap)
const SIZES_BINARY = [100, 500, 1000, 1500, 2000];   // max 2000: 3 × 192MB = 576MB
const SIZES_UNARY  = [100, 500, 1000, 2000, 3000];   // max 3000: 2 × 432MB = 864MB
const SIZES_STATS  = [100, 500, 1000, 2000, 3000];   // max 3000: 1 × 432MB + output
const SIZES_PROPS  = [100, 500, 1000, 2000];          // proprietà: spesso O(n²) con early-exit
const WARMUP = 2, REPS = 4;

// ── TIPI ────────────────────────────────────────────────────────────────────
interface SimpleEntry {
    group: string; operation: string; complexity: string;
    size: number; avgMs: number; minMs: number; stdMs: number;
    memMB: number; gbps?: number;
}
const results: SimpleEntry[] = [];

// ── MEASURE ─────────────────────────────────────────────────────────────────
function gc() { if ((globalThis as any).gc) (globalThis as any).gc(); }
function heapMB() { return process.memoryUsage().heapUsed / 1024 / 1024; }

function measure(fn: () => void, reps = REPS) {
    for (let i = 0; i < WARMUP; i++) { try { fn(); } catch {} }
    gc();
    const m0 = heapMB(), ts: number[] = [];
    for (let i = 0; i < reps; i++) {
        const t0 = performance.now(); try { fn(); } catch {}
        ts.push(performance.now() - t0);
    }
    const m1 = heapMB();
    const avg = ts.reduce((a, b) => a + b) / ts.length;
    const std = Math.sqrt(ts.reduce((a, b) => a + (b - avg) ** 2, 0) / ts.length);
    return { avg, min: Math.min(...ts), std, mem: Math.max(0, m1 - m0) };
}

function r3(x: number, d = 3) { return Math.round(x * 10 ** d) / 10 ** d; }

function rec(
    group: string, op: string, complexity: string, n: number,
    m: ReturnType<typeof measure>, byteFactor?: number
) {
    // throughput GB/s: bytes_letti_scritti / (tempo_secondi)
    const matBytes = n * n * 8;   // n² × 8 byte se fosse Float64Array nativa
    const gbps = byteFactor && m.avg > 0 ? (byteFactor * matBytes / 1e9) / (m.avg / 1000) : undefined;
    results.push({
        group, operation: op, complexity, size: n,
        avgMs: r3(m.avg), minMs: r3(m.min), stdMs: r3(m.std),
        memMB: r3(m.mem, 4), gbps: gbps !== undefined ? r3(gbps, 2) : undefined
    });
}

// ── BANNER ──────────────────────────────────────────────────────────────────
console.log("╔══════════════════════════════════════════════════════╗");
console.log("║   numeric-matrix — BENCHMARK SEMPLICE (grandi n)    ║");
console.log(`║   Node ${process.version.padEnd(10)}                              ║`);
console.log(`║   Binary:${SIZES_BINARY.join(",").padEnd(24)} Unary:${SIZES_UNARY.join(",").padEnd(18)}║`);
console.log("╚══════════════════════════════════════════════════════╝");
console.log("\nNota: Float64M wrappa ogni numero in un oggetto JS (~48 byte/elem).");
console.log("      n=2000 = 4M oggetti × 3 matrici = 576MB — limite sicuro operazioni binarie.\n");

// ── A: ARITMETICA BINARIA O(n²) ─────────────────────────────────────────────
console.log("━━━  A. Aritmetica element-wise (binaria)  ━━━━━━━━━━━━");
for (const n of SIZES_BINARY) {
    const A = Matrix.random(n, n), B = Matrix.random(n, n);
    const ops: [string, () => void, number][] = [
        ["add",        () => A.add(B),       3],
        ["sub",        () => A.sub(B),       3],
        ["dotMul",     () => A.dotMul(B),    3],
        ["dotDiv",     () => A.dotDiv(B),    3],
        ["mul scalar", () => A.mul(3.14),    2],
        ["dotPow(2)",  () => A.dotPow(2),    2],
    ];
    for (const [op, fn, bf] of ops) {
        const m = measure(fn);
        rec("Aritmetica", op, "O(n²)", n, m, bf);
    }
    const add = results.find(r=>r.operation==="add"&&r.size===n)!;
    console.log(`  n=${n.toString().padStart(5)}  add=${add.avgMs.toFixed(1).padStart(7)}ms  `+
        `dotMul=${results.find(r=>r.operation==="dotMul"&&r.size===n)!.avgMs.toFixed(1).padStart(7)}ms  `+
        `GB/s≈${add.gbps?.toFixed(1)??'?'} (teorico, se fosse Float64Array nativa)`);
}

// ── B: FUNZIONI UNARIE O(n²) ────────────────────────────────────────────────
console.log("\n━━━  B. Funzioni unarie  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
for (const n of SIZES_UNARY) {
    const A = Matrix.random(n, n);
    const Apos = A.abs();
    const ops: [string, () => void][] = [
        ["abs",    () => A.abs()],
        ["negate", () => A.negate()],
        ["round",  () => A.round()],
        ["sqrt",   () => Apos.sqrt()],
        ["sin",    () => A.sin()],
        ["cos",    () => A.cos()],
        ["exp",    () => A.mul(0.001).exp()],
    ];
    for (const [op, fn] of ops) {
        const m = measure(fn);
        rec("Unarie", op, "O(n²)", n, m, 2);
    }
    console.log(`  n=${n.toString().padStart(5)}  abs=${results.find(r=>r.operation==="abs"&&r.size===n)!.avgMs.toFixed(1).padStart(7)}ms  `+
        `sin=${results.find(r=>r.operation==="sin"&&r.size===n)!.avgMs.toFixed(1).padStart(7)}ms  `+
        `sqrt=${results.find(r=>r.operation==="sqrt"&&r.size===n)!.avgMs.toFixed(1).padStart(7)}ms`);
}

// ── C: TRASFORMAZIONI O(n²) ─────────────────────────────────────────────────
console.log("\n━━━  C. Trasformazioni  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
for (const n of SIZES_UNARY) {
    const A = Matrix.random(n, n);
    const ops: [string, () => void][] = [
        ["transpose", () => A.t()],
        ["ct",        () => A.ct()],
        ["flip ud",   () => A.flip(1)],
        ["flip lr",   () => A.flip(2)],
        ["rot90",     () => A.rot90()],
        ["clone",     () => A.clone()],
    ];
    for (const [op, fn] of ops) {
        const m = measure(fn);
        rec("Trasform.", op, "O(n²)", n, m, 2);
    }
    console.log(`  n=${n.toString().padStart(5)}  t=${results.find(r=>r.operation==="transpose"&&r.size===n)!.avgMs.toFixed(2).padStart(7)}ms  `+
        `rot90=${results.find(r=>r.operation==="rot90"&&r.size===n)!.avgMs.toFixed(2).padStart(7)}ms  `+
        `clone=${results.find(r=>r.operation==="clone"&&r.size===n)!.avgMs.toFixed(2).padStart(7)}ms`);
}

// ── D: STATISTICHE E NORME O(n²) ────────────────────────────────────────────
console.log("\n━━━  D. Statistiche e norme  ━━━━━━━━━━━━━━━━━━━━━━━━━");
for (const n of SIZES_STATS) {
    const A = Matrix.random(n, n);
    const ops: [string, string, () => void][] = [
        ["totalSum",  "O(n²)", () => A.totalSum()],
        ["sum col",   "O(n²)", () => A.sum(1)],
        ["sum row",   "O(n²)", () => A.sum(2)],
        ["mean",      "O(n²)", () => A.mean(1)],
        ["max col",   "O(n²)", () => A.max(1)],
        ["min col",   "O(n²)", () => A.min(1)],
        ["norm Fro",  "O(n²)", () => A.norm("Fro")],
        ["norm 1",    "O(n²)", () => A.norm("1")],
        ["norm Inf",  "O(n²)", () => A.norm("Inf")],
        ["trace",     "O(n)",  () => A.trace()],
    ];
    for (const [op, cplx, fn] of ops) {
        const m = measure(fn);
        rec("Statistiche", op, cplx, n, m);
    }
    console.log(`  n=${n.toString().padStart(5)}  totalSum=${results.find(r=>r.operation==="totalSum"&&r.size===n)!.avgMs.toFixed(2).padStart(7)}ms  `+
        `normFro=${results.find(r=>r.operation==="norm Fro"&&r.size===n)!.avgMs.toFixed(2).padStart(7)}ms  `+
        `trace=${results.find(r=>r.operation==="trace"&&r.size===n)!.avgMs.toFixed(3).padStart(7)}ms`);
}

// ── E: PROPRIETÀ STRUTTURALI ────────────────────────────────────────────────
console.log("\n━━━  E. Proprietà strutturali  ━━━━━━━━━━━━━━━━━━━━━━━━");
for (const n of SIZES_PROPS) {
    const A    = Matrix.random(n, n);
    const Asym = A.add(A.t()).mul(0.5) as Matrix<Float64M>;
    const U    = Matrix.triu(A);
    const L    = Matrix.tril(A);
    const Z    = Matrix.zeros(n, n);

    const props: [string, string, () => void][] = [
        ["isSquare",             "O(1)",  () => A.isSquare()],
        ["isSymmetric",          "O(n²)", () => Asym.isSymmetric()],
        ["isUpperTriangular",    "O(n²)", () => U.isUpperTriangular()],   // early-exit: primo elem ≠ 0 sotto diag
        ["isLowerTriangular",    "O(n²)", () => L.isLowerTriangular()],
        ["isDiagonal",           "O(n²)", () => Matrix.diag(n,1).isDiagonal()],
        ["isZeroMatrix",         "O(n²)", () => Z.isZeroMatrix()],
        ["hasFiniteValues",      "O(n²)", () => A.hasFiniteValues()],
        ["isDiagonallyDominant", "O(n²)", () => A.isDiagonallyDominant()],
        ["hasZeroTrace",         "O(n)",  () => A.hasZeroTrace()],
    ];
    for (const [op, cplx, fn] of props) {
        const m = measure(fn);
        rec("Proprietà", op, cplx, n, m);
    }
    console.log(`  n=${n.toString().padStart(5)}  isSymmetric=${results.find(r=>r.operation==="isSymmetric"&&r.size===n)!.avgMs.toFixed(2).padStart(6)}ms  `+
        `isDiagDom=${results.find(r=>r.operation==="isDiagonallyDominant"&&r.size===n)!.avgMs.toFixed(2).padStart(6)}ms  `+
        `hasFinite=${results.find(r=>r.operation==="hasFiniteValues"&&r.size===n)!.avgMs.toFixed(2).padStart(6)}ms`);
}

// ── F: SCALABILITÀ TRASPOSIZIONE (memory-bandwidth puro) ────────────────────
console.log("\n━━━  F. Scalabilità trasposizione — memory bandwidth  ━━━");
console.log("  (dovrebbe scalare ~O(n²), throughput ~costante se memory-bound)");
console.log("  (il throughput reale è basso perché ogni elemento è un oggetto JS)");
for (const n of SIZES_UNARY) {
    const A = Matrix.random(n, n);
    const m = measure(() => A.t(), REPS);
    // Calcola throughput teorico vs reale
    // Teorico (Float64Array): 2 × n² × 8 bytes
    // Reale (Float64M[]): 2 × n² × 48 bytes circa
    const gbps_theoretical = (2 * n * n * 8 / 1e9) / (m.avg / 1000);
    const gbps_actual      = (2 * n * n * 48 / 1e9) / (m.avg / 1000);
    rec("Scalabilità t", "transpose", "O(n²)", n, m, 2);
    console.log(`  n=${n.toString().padStart(6)}  ${m.avg.toFixed(1).padStart(8)}ms  `+
        `GB/s teorico=${gbps_theoretical.toFixed(2).padStart(5)}  reale≈${gbps_actual.toFixed(2).padStart(5)}`);
}

// ── G: OVERHEAD OGGETTI — confronto accesso diretto ─────────────────────────
console.log("\n━━━  G. Overhead oggetti Float64M: add vs Float64Array nativa  ━━━");
console.log("  (mostra il costo dell'astrazione generica)");
for (const n of SIZES_BINARY) {
    const A = Matrix.random(n, n), B = Matrix.random(n, n);
    // Tempo add Float64M (con oggetti)
    const mWrapped = measure(() => A.add(B));
    // Tempo add Float64Array nativa (senza oggetti, solo numeri)
    const aRaw = new Float64Array(n * n);
    const bRaw = new Float64Array(n * n);
    const cRaw = new Float64Array(n * n);
    for (let i = 0; i < n*n; i++) { aRaw[i] = Math.random(); bRaw[i] = Math.random(); }
    const mNative = measure(() => { for (let i=0;i<n*n;i++) cRaw[i]=aRaw[i]+bRaw[i]; }, REPS);
    const overhead = mWrapped.avg / mNative.avg;
    rec("Overhead", "add Float64M",      "O(n²)", n, mWrapped);
    rec("Overhead", "add Float64Array",  "O(n²)", n, mNative);
    console.log(`  n=${n.toString().padStart(5)}  F64M=${mWrapped.avg.toFixed(1).padStart(7)}ms  NativeArr=${mNative.avg.toFixed(1).padStart(6)}ms  overhead=×${overhead.toFixed(1)}`);
}

// ── SAVE JSON ────────────────────────────────────────────────────────────────
writeFileSync("benchmark/benchmark_simple_results.json",
    JSON.stringify({ meta: { date: new Date().toISOString(), node: process.version, note: "Float64M object overhead benchmark. See docs for size limits." }, results }, null, 2));

// ── HTML ─────────────────────────────────────────────────────────────────────

const groups: Record<string, { ops: string[]; sizes: number[]; pal: string[] }> = {
    "Aritmetica": { ops: ["add","sub","dotMul","dotDiv","mul scalar","dotPow(2)"],
        sizes: SIZES_BINARY, pal: ["#1565C0","#1976D2","#1E88E5","#42A5F5","#0097A7","#006064"] },
    "Unarie":     { ops: ["abs","negate","round","sqrt","sin","cos","exp"],
        sizes: SIZES_UNARY,  pal: ["#B71C1C","#C62828","#D32F2F","#E53935","#EF5350","#FF8A80","#FFCDD2"] },
    "Trasform.":  { ops: ["transpose","ct","flip ud","flip lr","rot90","clone"],
        sizes: SIZES_UNARY,  pal: ["#1B5E20","#2E7D32","#388E3C","#43A047","#66BB6A","#A5D6A7"] },
    "Statistiche":{ ops: ["totalSum","sum col","sum row","mean","norm Fro","norm 1","norm Inf","trace"],
        sizes: SIZES_STATS,  pal: ["#E65100","#EF6C00","#F57C00","#FB8C00","#FFA726","#FFB74D","#FFCC80","#FF8A65"] },
    "Proprietà":  { ops: ["isSymmetric","isUpperTriangular","isLowerTriangular","isDiagonal","isZeroMatrix","hasFiniteValues","isDiagonallyDominant","hasZeroTrace"],
        sizes: SIZES_PROPS,  pal: ["#4A148C","#6A1B9A","#7B1FA2","#8E24AA","#AB47BC","#CE93D8","#E1BEE7","#9C27B0"] },
};

function makeCharts(): string {
    return Object.entries(groups).map(([grp, { ops, sizes, pal }]) => {
        const labels = JSON.stringify(sizes.map(n => n.toLocaleString()));
        const ds = ops.map((op, ci) => {
            const pts = sizes.map(n => results.find(r => r.group === grp && r.operation === op && r.size === n)?.avgMs ?? null);
            return `{label:${JSON.stringify(op)},data:${JSON.stringify(pts)},borderColor:"${pal[ci]}",backgroundColor:"${pal[ci]}22",tension:.3,pointRadius:4,fill:false}`;
        });
        const id = `c_${grp.replace(/[^a-z]/gi,"_")}`;
        return `
<section>
  <h2>${grp} — scalabilità Float64M (grandi n)</h2>
  <div class="chart-wrap"><canvas id="${id}"></canvas></div>
</section>
<script>
new Chart(document.getElementById("${id}"),{type:"line",
  data:{labels:${labels},datasets:[${ds.join(",")}]},
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{position:"bottom",labels:{boxWidth:10,font:{size:11}}}},
    scales:{y:{title:{display:true,text:"ms"},beginAtZero:true},x:{title:{display:true,text:"n"}}}}
});</script>`;
    }).join("\n");
}

// Overhead chart
const overheadRows = SIZES_BINARY.map(n => ({
    n,
    wrapped:  results.find(r => r.group === "Overhead" && r.operation === "add Float64M" && r.size === n)?.avgMs ?? 0,
    native:   results.find(r => r.group === "Overhead" && r.operation === "add Float64Array" && r.size === n)?.avgMs ?? 0,
}));

const allRows = [...results].sort((a, b) => a.group.localeCompare(b.group) || a.size - b.size)
    .map(r => `<tr><td>${r.group}</td><td>${r.operation}</td><td>${r.complexity}</td><td>${r.size.toLocaleString()}</td><td>${r.avgMs}</td><td>${r.minMs}</td><td>${r.stdMs}</td><td>${r.memMB}</td><td>${r.gbps ?? "—"}</td></tr>`)
    .join("\n");

const html = `<!DOCTYPE html><html lang="it"><head><meta charset="UTF-8">
<title>Benchmark Semplice — numeric-matrix</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f0f2f5;color:#222;font-size:13px}
header{background:#1B5E20;color:#fff;padding:18px 32px}
header h1{font-size:1.4rem;font-weight:600}
header p{opacity:.75;font-size:.8rem;margin-top:4px}
main{max-width:1400px;margin:0 auto;padding:16px}
section{background:#fff;border-radius:8px;padding:14px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
h2{font-size:.9rem;font-weight:600;color:#1B5E20;margin-bottom:10px;border-bottom:2px solid #E8F5E9;padding-bottom:5px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.chart-wrap{position:relative;height:280px}
.chart-wrap-lg{position:relative;height:360px}
table{width:100%;border-collapse:collapse;font-size:.74rem}
th{background:#E8F5E9;color:#1B5E20;padding:4px 7px;text-align:left;border-bottom:2px solid #A5D6A7}
td{padding:3px 7px;border-bottom:1px solid #eee}
tr:hover td{background:#F1F8F1}
.warn{background:#FFF9C4;border-left:3px solid #F9A825;padding:10px 14px;border-radius:4px;margin-bottom:14px;font-size:.8rem;line-height:1.5}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;margin-bottom:14px}
.card{background:#E8F5E9;border-radius:6px;padding:8px 12px}
.card .val{font-size:1.3rem;font-weight:700;color:#1B5E20}
.card .lbl{font-size:.65rem;color:#546E7A;margin-top:2px}
</style></head><body>
<header>
  <h1>numeric-matrix — Benchmark Operazioni Semplici</h1>
  <p>${new Date().toLocaleString("it-IT")} · Node ${process.version} · Float64M only</p>
</header>
<main>
<div class="warn">
  <strong>⚠️ Overhead oggetti:</strong>
  Matrix&lt;Float64M&gt; memorizza ogni elemento come oggetto JS separato (~48 byte/elem).
  Un'operazione <code>add</code> su n×n crea n² nuovi oggetti Float64M allocati nell'heap V8.
  A n=2000 questo produce 4M oggetti × 48B = 192MB <em>per matrice</em>, contro i 32MB di una Float64Array nativa.
  Il throughput "GB/s" mostrato è calcolato su <em>8 byte/elem</em> (valore nativo), quindi è il throughput <em>teorico massimo</em> se usassimo Float64Array — il throughput reale è ~6× inferiore per l'overhead oggetti.
  Per n &gt; 2000 con operazioni binarie, passare <code>--max-old-space-size=8192</code>.
</div>

<div class="grid3">
  <div class="card"><div class="val">${results.length}</div><div class="lbl">Misurazioni</div></div>
  <div class="card"><div class="val">${[...new Set(results.map(r=>r.size))].length}</div><div class="lbl">Dimensioni</div></div>
  <div class="card"><div class="val">~${overheadRows.length>0?Math.round(overheadRows.reduce((s,r)=>s+(r.wrapped/r.native),0)/overheadRows.length):0}×</div><div class="lbl">Overhead medio vs Float64Array</div></div>
  <div class="card"><div class="val">${Math.max(...SIZES_BINARY).toLocaleString()}</div><div class="lbl">n max (binarie)</div></div>
  <div class="card"><div class="val">${Math.max(...SIZES_UNARY).toLocaleString()}</div><div class="lbl">n max (unarie)</div></div>
</div>

<section>
  <h2>Overhead Float64M vs Float64Array nativa — add(A,B)</h2>
  <div class="chart-wrap"><canvas id="c_overhead"></canvas></div>
</section>

<div class="grid2">
${makeCharts()}
</div>

<section>
  <h2>Tabella completa</h2>
  <table><thead><tr><th>Gruppo</th><th>Operazione</th><th>Comp.</th><th>n</th><th>Avg ms</th><th>Min ms</th><th>σ ms</th><th>Mem MB</th><th>GB/s*</th></tr></thead>
  <tbody>${allRows}</tbody></table>
  <p style="font-size:.7rem;color:#888;margin-top:6px">*GB/s calcolato su 8 byte/elem (Float64 nativo). Il throughput reale con gli oggetti JS è ~6× inferiore.</p>
</section>
</main>
<script>
// Overhead chart
new Chart(document.getElementById("c_overhead"),{type:"bar",
  data:{
    labels:${JSON.stringify(SIZES_BINARY.map(n=>n.toLocaleString()))},
    datasets:[
      {label:"Float64M add (oggetti JS)",     data:${JSON.stringify(overheadRows.map(r=>r.wrapped))}, backgroundColor:"#E91E63CC"},
      {label:"Float64Array add (nativa raw)", data:${JSON.stringify(overheadRows.map(r=>r.native))},  backgroundColor:"#2196F3CC"},
    ]
  },
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{position:"bottom"}},
    scales:{y:{title:{display:true,text:"ms (log)"},type:"logarithmic"},x:{title:{display:true,text:"n"}}}}
});
</script>
</body></html>`;

writeFileSync("benchmark/benchmark_simple_report.html", html);
console.log("\n  ✓ benchmark/benchmark_simple_results.json");
console.log("  ✓ benchmark/benchmark_simple_report.html");
console.log("\n  Apri benchmark/benchmark_simple_report.html nel browser.");
console.log("  Per n più grandi: npx tsx --expose-gc --max-old-space-size=8192 benchmark/benchmark_simple.ts");

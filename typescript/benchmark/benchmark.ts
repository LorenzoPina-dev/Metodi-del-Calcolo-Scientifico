// benchmark/benchmark.ts
// Utilizzo: npx tsx --expose-gc benchmark/benchmark.ts

import { writeFileSync } from "node:fs";
import { performance }   from "node:perf_hooks";
import { Matrix, Float64M, Complex, Rational } from "../src/index.js";
import { solveSOR } from "../src/solver/sor.js";
import { solveCG }  from "../src/solver/cg.js";

// ── CONFIG ───────────────────────────────────────────────────
const CFG = {
    warmup:    3,
    reps:      5,
    small:     [10, 25, 50, 100, 200],
    medium:    [10, 50, 100, 200, 400, 600],
    rationalN: [3, 4, 5, 6],
};

interface BenchEntry {
    category: string; operation: string; size: number; type: string;
    avgMs: number; minMs: number; maxMs: number; stdMs: number;
    memDeltaMB: number; residual?: number;
}
const results: BenchEntry[] = [];

// ── MISURA ───────────────────────────────────────────────────
function heapMB() { return process.memoryUsage().heapUsed / 1024 / 1024; }
function gc() { if ((globalThis as any).gc) (globalThis as any).gc(); }

function measure(fn: () => void, reps = CFG.reps) {
    for (let i = 0; i < CFG.warmup; i++) { try { fn(); } catch {} }
    gc();
    const m0 = heapMB(), ts: number[] = [];
    for (let i = 0; i < reps; i++) {
        const t0 = performance.now(); try { fn(); } catch {}
        ts.push(performance.now() - t0);
    }
    const m1 = heapMB();
    const avg = ts.reduce((a, b) => a + b) / ts.length;
    const std = Math.sqrt(ts.reduce((a, b) => a + (b - avg) ** 2, 0) / ts.length);
    return { avgMs: avg, minMs: Math.min(...ts), maxMs: Math.max(...ts), stdMs: std, memDeltaMB: Math.max(0, m1 - m0) };
}

function rec(cat: string, op: string, n: number, type: string, m: ReturnType<typeof measure>, res?: number) {
    const r3 = (x: number, d = 3) => Math.round(x * 10 ** d) / 10 ** d;
    results.push({ category: cat, operation: op, size: n, type,
        avgMs: r3(m.avgMs), minMs: r3(m.minMs), maxMs: r3(m.maxMs), stdMs: r3(m.stdMs),
        memDeltaMB: r3(m.memDeltaMB, 4), residual: res });
}

// ── HELPER MATRICI ───────────────────────────────────────────
function makeDiagDom(n: number): Matrix<Float64M> {
    const A = Matrix.random(n, n);
    for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) if (i !== j) s += Math.abs(A.get(i, j).value);
        A.set(i, i, s + 1 + Math.random());
    }
    return A;
}
function makeSPD(n: number): Matrix<Float64M> {
    const M = Matrix.random(n, n);
    return M.t().mul(M).add(Matrix.identity(n).mul(n * 0.01)) as Matrix<Float64M>;
}
/** SPD E diagonalmente dominante: garantisce convergenza degli iterativi */
function makeSPDdiagDom(n: number): Matrix<Float64M> {
    const A = Matrix.random(n, n);
    const S = A.add(A.t()).mul(0.5) as Matrix<Float64M>;  // simmetrizza
    for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) if (i !== j) s += Math.abs(S.get(i, j).value);
        S.set(i, i, s + 1 + Math.random());
    }
    return S;
}
function makeComplexMat(n: number): Matrix<Complex> {
    const A = Matrix.zerosOf(n, n, Complex.zero, Complex.one);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) A.set(i, j, new Complex((Math.random()-.5)*2,(Math.random()-.5)*2));
        A.set(i, i, new Complex(n * 2, 0));
    }
    return A;
}
function makeHilbert(n: number): Matrix<Rational> {
    const H = Matrix.zerosOf(n, n, Rational.zero, Rational.one);
    for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) H.set(i, j, new Rational(1, i+j+1));
    return H;
}
function relRes<T extends { kind: string; toNumber(): number; subtract(o: any): any; abs(): any; }>(
    A: Matrix<T>, x: Matrix<T>, b: Matrix<T>
): number {
    const Ax = A.mul(x);
    let nR=0, nA=0, nX=0, nB=0;
    for (let i=0; i<b.rows; i++) {
        const r = (Ax.get(i,0) as any).subtract(b.get(i,0)).abs().toNumber();
        if (r>nR) nR=r;
        const bv = (b.get(i,0) as any).abs().toNumber(); if(bv>nB) nB=bv;
        const xv = (x.get(i,0) as any).abs().toNumber(); if(xv>nX) nX=xv;
    }
    for (let i=0; i<A.rows; i++) { let s=0; for(let j=0; j<A.cols; j++) s+=(A.get(i,j) as any).abs().toNumber(); if(s>nA) nA=s; }
    const d = nA*nX+nB; return d<1e-300 ? 0 : nR/d;
}

// ── SEZ 1: CORE F64 ──────────────────────────────────────────
function benchFloat64() {
    console.log("\n━━━  1. FLOAT64M — operazioni core  ━━━━━━━━━━━━━━━━━━");
    for (const n of CFG.medium) {
        const A=makeDiagDom(n), B=Matrix.random(n,n);
        const ops: [string, string, ()=>void][] = [
            ["Aritmetica","add",()=>A.add(B)], ["Aritmetica","sub",()=>A.sub(B)],
            ["Aritmetica","mul",()=>A.mul(B)], ["Aritmetica","mul scalar",()=>A.mul(3.14)],
            ["Aritmetica","pow(2)",()=>A.pow(2)],
            ["DotOps","dotMul",()=>A.dotMul(B)], ["DotOps","dotDiv",()=>A.dotDiv(B)],
            ["DotOps","dotPow(2)",()=>A.dotPow(2)],
            ["Unarie","abs",()=>A.abs()], ["Unarie","sqrt",()=>A.sqrt()],
            ["Unarie","round",()=>A.round()], ["Unarie","negate",()=>A.negate()],
            ["Unarie","exp",()=>A.abs().mul(0.001).exp()],
            ["Unarie","sin",()=>A.sin()], ["Unarie","cos",()=>A.cos()],
            ["Trasform.","transpose",()=>A.t()], ["Trasform.","ct",()=>A.ct()],
            ["Trasform.","flip ud",()=>A.flip(1)], ["Trasform.","flip lr",()=>A.flip(2)],
            ["Trasform.","rot90",()=>A.rot90()], ["Trasform.","repmat 2x2",()=>A.repmat(2,2)],
            ["Stat.","sum col",()=>A.sum(1)], ["Stat.","sum row",()=>A.sum(2)],
            ["Stat.","mean",()=>A.mean(1)], ["Stat.","max",()=>A.max(1)],
            ["Stat.","totalSum",()=>A.totalSum()],
            ["Stat.","norm Fro",()=>A.norm("Fro")], ["Stat.","norm 1",()=>A.norm("1")],
            ["Stat.","norm Inf",()=>A.norm("Inf")], ["Algebra","trace",()=>A.trace()],
        ];
        for (const [c,o,f] of ops) rec(c,o,n,"Float64M",measure(f));
        const mulT = results.find(r=>r.operation==="mul"&&r.size===n&&r.type==="Float64M")?.avgMs.toFixed(2);
        const tT   = results.find(r=>r.operation==="transpose"&&r.size===n)?.avgMs.toFixed(3);
        console.log(`  n=${n.toString().padStart(4)}  mul=${String(mulT).padStart(7)}ms  t=${String(tT).padStart(7)}ms`);
    }
}

// ── SEZ 2: DECOMPOSIZIONI ────────────────────────────────────
function benchDecompositions() {
    console.log("\n━━━  2. DECOMPOSIZIONI — Float64M  ━━━━━━━━━━━━━━━━━━━");
    for (const n of CFG.small) {
        const A=makeDiagDom(n), Aspd=makeSPD(n);
        const decomps: [string,()=>void][] = [
            ["LUP",      ()=>Matrix.decomp.lup(A)],
            ["LU",       ()=>Matrix.decomp.lu(A)],
            ["LU total", ()=>Matrix.decomp.lu_total(A)],
            ["QR",       ()=>Matrix.decomp.qr(A)],
            ["Cholesky", ()=>Matrix.decomp.cholesky(Aspd)],
            ["LDLT",     ()=>Matrix.decomp.ldlt(Aspd)],
            ["tril",     ()=>Matrix.tril(A)],
            ["triu",     ()=>Matrix.triu(A)],
        ];
        for (const [op,fn] of decomps) rec("Decomp.",op,n,"Float64M",measure(fn));
        const g=(op:string)=>results.filter(r=>r.operation===op&&r.size===n).at(-1)?.avgMs.toFixed(2);
        console.log(`  n=${n.toString().padStart(4)}  LUP=${g("LUP")}ms  QR=${g("QR")}ms  Chol=${g("Cholesky")}ms  LDLT=${g("LDLT")}ms`);
    }
}

// ── SEZ 3: SOLVER ────────────────────────────────────────────
function benchSolvers() {
    console.log("\n━━━  3. SOLVER — Float64M (residuo relativo)  ━━━━━━━━");
    for (const n of CFG.small) {
        // Matrici con proprietà chiare
        const Addom  = makeDiagDom(n);       // diag. dominante → tutti gli iterativi convergono
        const Aspd   = makeSPD(n);           // SPD → Cholesky, CG
        const b      = Matrix.random(n, 1);
        const bDdom  = Addom.mul(b);
        const bSpd   = Aspd.mul(b);

        const solvers: [string, Matrix<Float64M>, Matrix<Float64M>, string][] = [
            ["LUP",          Addom, bDdom, "LUP"],
            ["LU",           Addom, bDdom, "LU"],
            ["QR",           Addom, bDdom, "QR"],
            ["Cholesky",     Aspd,  bSpd,  "CHOLESKY"],
            ["LDLT",         Aspd,  bSpd,  "LDLT"],
            ["Jacobi",       Addom, bDdom, "JACOBI"],
            ["Gauss-Seidel", Addom, bDdom, "GAUSS-SEIDEL"],
            ["SOR ω=1.0",    Addom, bDdom, "SOR_1.0"],
            ["SOR ω=1.5",    Addom, bDdom, "SOR_1.5"],
            ["CG (SPD)",     Aspd,  bSpd,  "CG"],
        ];

        for (const [label, mat, rhs, method] of solvers) {
            let residual = NaN;
            const realMethod = method.startsWith("SOR") ? "SOR" : method;
            const m = measure(() => {
                try {
                    const x = method === "SOR_1.0" ? solveSOR(mat, rhs, 1.0)
                            : method === "SOR_1.5" ? solveSOR(mat, rhs, 1.5)
                            : mat.solve(rhs, realMethod);
                    residual = relRes(mat, x, rhs);
                } catch {}
            });
            rec("Solver", label, n, "Float64M", m, isNaN(residual) ? undefined : residual);
            const sym = residual<1e-8?"✓":residual<1e-4?"~":isNaN(residual)?"err":"✗";
            console.log(`  n=${n.toString().padStart(4)}  ${label.padEnd(16)} ${m.avgMs.toFixed(3).padStart(9)}ms  res=${isNaN(residual)?"N/A  ":residual.toExponential(2)} ${sym}`);
        }
        console.log();
    }
}

// ── SEZ 4: SCALABILITÀ mul ───────────────────────────────────
function benchMulScaling() {
    console.log("\n━━━  4. SCALABILITÀ mul — Float64M  ━━━━━━━━━━━━━━━━━━");
    for (const n of [10,25,50,100,150,200,300,400,500]) {
        const A=Matrix.random(n,n), B=Matrix.random(n,n);
        const m=measure(()=>A.mul(B),3);
        const gflops=(2*n**3)/(m.avgMs*1e6);
        rec("Scalabilità","mul A×B",n,"Float64M",m);
        console.log(`  n=${n.toString().padStart(4)}  ${m.avgMs.toFixed(2).padStart(8)}ms  ~${gflops.toFixed(3)} GFLOPS`);
    }
}

// ── SEZ 5: ITERATIVI convergenza ─────────────────────────────
function benchIterativeConvergence() {
    console.log("\n━━━  5. SOLVER ITERATIVI — analisi convergenza  ━━━━━━");
    const n = 50;
    // IMPORTANTE: usa matrice SPD E diagonalmente dominante
    // così sia gli statici (Jacobi/GS/SOR) sia CG convergono
    const A = makeSPDdiagDom(n);
    const b = Matrix.random(n, 1);
    const bA = A.mul(b);

    console.log(`  Matrice: SPD + diag. dominante, n=${n}`);
    console.log(`  isDiagDom=${A.isDiagonallyDominant()}, isSymmetric=${A.isSymmetric()}`);

    // SOR vs omega
    console.log("  SOR — vari ω:");
    for (const omega of [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 1.9]) {
        let res=NaN;
        const m=measure(()=>{ try { const x=solveSOR(A,bA,omega); res=relRes(A,x,bA); } catch {} });
        rec("Iterativi",`SOR ω=${omega}`,n,"Float64M",m,isNaN(res)?undefined:res);
        const sym=res<1e-8?"✓":res<1e-4?"~":isNaN(res)?"err":"✗";
        console.log(`    ω=${omega.toFixed(1)}  ${m.avgMs.toFixed(3).padStart(8)}ms  res=${isNaN(res)?"N/A":res.toExponential(2)} ${sym}`);
    }

    // Confronto diretto tutti i metodi
    console.log("  Confronto diretto:");
    const cmp: [string,()=>Matrix<Float64M>][] = [
        ["Jacobi",       ()=>A.solve(bA,"JACOBI")],
        ["Gauss-Seidel", ()=>A.solve(bA,"GAUSS-SEIDEL")],
        ["SOR ω=1.0",    ()=>solveSOR(A,bA,1.0)],
        ["SOR ω=1.5",    ()=>solveSOR(A,bA,1.5)],
        ["CG",           ()=>solveCG(A,bA)],
        ["Cholesky",     ()=>A.solve(bA,"CHOLESKY")],
        ["LUP",          ()=>A.solve(bA,"LUP")],
    ];
    for (const [lbl,fn] of cmp) {
        let res=NaN;
        const m=measure(()=>{ try { const x=fn(); res=relRes(A,x,bA); } catch {} });
        rec("Iterativi vs Diretti",lbl,n,"Float64M",m,isNaN(res)?undefined:res);
        const sym=res<1e-8?"✓":res<1e-4?"~":isNaN(res)?"err":"✗";
        console.log(`    ${lbl.padEnd(16)} ${m.avgMs.toFixed(3).padStart(8)}ms  res=${isNaN(res)?"N/A":res.toExponential(2)} ${sym}`);
    }
}

// ── SEZ 6: TUTTE LE PROPRIETÀ ────────────────────────────────
function benchProperties() {
    console.log("\n━━━  6. PROPRIETÀ STRUTTURALI — Float64M (tutte)  ━━━━");
    for (const n of [50, 100, 200, 400]) {
        const A    = Matrix.random(n, n);
        const Asym = A.add(A.t()).mul(0.5) as Matrix<Float64M>;
        const Addom = makeDiagDom(n);
        const I    = Matrix.identity(n);
        const U    = Matrix.triu(A);
        const L    = Matrix.tril(A);
        const D    = Matrix.diag(n, 1.0);
        const Z    = Matrix.zeros(n, n);
        const Aspd = makeSPD(n);
        // matrice stocastica (colonne sommano a 1)
        const Stoch = (() => {
            const S = Matrix.random(n, n).abs();
            for (let j=0; j<n; j++) {
                let s=0; for (let i=0; i<n; i++) s += S.get(i,j).value;
                for (let i=0; i<n; i++) S.set(i,j, S.get(i,j).value/s);
            }
            return S;
        })();

        const props: [string, ()=>void][] = [
            ["isSquare",              ()=>A.isSquare()],
            ["isSymmetric",           ()=>Asym.isSymmetric()],
            ["isUpperTriangular",     ()=>U.isUpperTriangular()],
            ["isLowerTriangular",     ()=>L.isLowerTriangular()],
            ["isDiagonal",            ()=>D.isDiagonal()],
            ["isIdentity",            ()=>I.isIdentity()],
            ["isOrthogonal",          ()=>I.isOrthogonal()],
            ["isZeroMatrix",          ()=>Z.isZeroMatrix()],
            ["isInvertible",          ()=>Addom.isInvertible()],
            ["isSingular",            ()=>Z.isSingular()],
            ["isPositiveDefinite",    ()=>Aspd.isPositiveDefinite()],
            ["isPositiveSemiDef",     ()=>Aspd.isPositiveSemiDefinite()],
            ["isDiagonallyDominant",  ()=>Addom.isDiagonallyDominant()],
            ["hasZeroTrace",          ()=>A.hasZeroTrace()],
            ["hasFiniteValues",       ()=>A.hasFiniteValues()],
            ["isStochastic",          ()=>Stoch.isStochastic()],
        ];
        for (const [op,fn] of props) rec("Proprietà",op,n,"Float64M",measure(fn));
        const g=(op:string)=>results.filter(r=>r.operation===op&&r.size===n&&r.type==="Float64M").at(-1)?.avgMs.toFixed(3);
        console.log(`  n=${n.toString().padStart(4)}  isSymmetric=${g("isSymmetric")}ms  isOrthogonal=${g("isOrthogonal")}ms  isInvertible=${g("isInvertible")}ms  isPosDef=${g("isPositiveDefinite")}ms`);
    }
}

// ── SEZ 7: COMPLEX ───────────────────────────────────────────
function benchComplex() {
    console.log("\n━━━  7. COMPLEX — operazioni e solver  ━━━━━━━━━━━━━━━");
    for (const n of [5,10,25,50,100]) {
        const A=makeComplexMat(n), B=makeComplexMat(n);
        const b=Matrix.zerosOf(n,1,Complex.zero,Complex.one);
        for (let i=0;i<n;i++) b.set(i,0,new Complex(Math.random(),Math.random()));
        const rhs=A.mul(b);
        const ops: [string,string,()=>void][] = [
            ["Aritm.","add (C)",()=>A.add(B)], ["Aritm.","mul (C)",()=>A.mul(B)],
            ["Trasf.","ct (C)",()=>A.ct()], ["Stat.","norm Fro (C)",()=>A.norm("Fro")],
            ["Algebra","det (C)",()=>A.det()],
            ["Decomp.","LUP (C)",()=>Matrix.decomp.lup(A)],
            ["Decomp.","QR (C)",()=>Matrix.decomp.qr(A)],
        ];
        for (const [c,o,f] of ops) rec(c,o,n,"Complex",measure(f));
        let rLUP=NaN, rQR=NaN;
        const mLUP=measure(()=>{ const x=A.solve(rhs,"LUP"); rLUP=relRes(A,x,rhs); });
        const mQR =measure(()=>{ const x=A.solve(rhs,"QR");  rQR =relRes(A,x,rhs); });
        rec("Solver","LUP (C)",n,"Complex",mLUP,rLUP);
        rec("Solver","QR (C)", n,"Complex",mQR, rQR);
        console.log(`  n=${n.toString().padStart(4)}  mul=${results.filter(r=>r.operation==="mul (C)"&&r.size===n).at(-1)?.avgMs.toFixed(3)}ms  LUP=${mLUP.avgMs.toFixed(3)}ms${rLUP<1e-8?" ✓":" ✗"}  QR=${mQR.avgMs.toFixed(3)}ms${rQR<1e-8?" ✓":" ✗"}`);
    }
}

// ── SEZ 8: RATIONAL ─────────────────────────────────────────
function benchRational() {
    console.log("\n━━━  8. RATIONAL — Hilbert n×n  ━━━━━━━━━━━━━━━━━━━━━");
    for (const n of CFG.rationalN) {
        const H=makeHilbert(n);
        const ones=Matrix.zerosOf(n,1,Rational.zero,Rational.one);
        for (let i=0;i<n;i++) ones.set(i,0,Rational.one);
        const mMul=measure(()=>H.mul(H),3);
        let det: Rational|null=null;
        const mDet=measure(()=>{ det=H.det() as Rational; },3);
        const rhs=H.mul(ones);
        let exact=false;
        const mSolve=measure(()=>{
            const x=H.solve(rhs,"LUP");
            const Hx=H.mul(x);
            exact=true;
            for (let i=0;i<n;i++) if ((Hx.get(i,0) as Rational).subtract(rhs.get(i,0) as Rational).num!==0n){exact=false;break;}
        },3);
        rec("Aritm.","mul (R)",n,"Rational",mMul);
        rec("Algebra","det (R)",n,"Rational",mDet);
        rec("Solver","LUP (R)",n,"Rational",mSolve,exact?0:1e-10);
        const x=H.solve(rhs,"LUP");
        let maxDen=0;
        for (let i=0;i<n;i++) { const bits=(x.get(i,0) as Rational).den.toString(2).length; if(bits>maxDen) maxDen=bits; }
        console.log(`  n=${n.toString().padStart(3)}  mul=${mMul.avgMs.toFixed(2)}ms  det=${mDet.avgMs.toFixed(2)}ms [${det?.toString()??'?'}]  solve=${mSolve.avgMs.toFixed(2)}ms  exact=${exact?"✓":"✗"}  max_den_bits=${maxDen}`);
    }
}

// ── SEZ 9: CONFRONTO TIPI ────────────────────────────────────
function benchTypeComparison() {
    console.log("\n━━━  9. CONFRONTO TIPI — mul e LUP  ━━━━━━━━━━━━━━━━━");
    for (const n of [5,10,20,30]) {
        const Af=makeDiagDom(n), Bf=Matrix.random(n,n);
        const Ac=makeComplexMat(n), Bc=makeComplexMat(n);
        const Ar=makeHilbert(n), Br=makeHilbert(n);
        const mf=measure(()=>Af.mul(Bf),5);
        const mc=measure(()=>Ac.mul(Bc),5);
        const mr=measure(()=>Ar.mul(Br),3);
        rec("Confronto","mul F64",n,"Float64M",mf);
        rec("Confronto","mul Cmx",n,"Complex",mc);
        rec("Confronto","mul Rat",n,"Rational",mr);
        const rFC=mc.avgMs/mf.avgMs, rFR=mr.avgMs/mf.avgMs;
        console.log(`  n=${n.toString().padStart(3)}  F64=${mf.avgMs.toFixed(3)}ms  Cmx=${mc.avgMs.toFixed(3)}ms(×${rFC.toFixed(1)})  Rat=${mr.avgMs.toFixed(3)}ms(×${rFR.toFixed(0)})`);
    }
}

// ── SEZ 10: GALLERY ──────────────────────────────────────────
function benchGallery() {
    console.log("\n━━━  10. GALLERY — costruzione n=100  ━━━━━━━━━━━━━━━━");
    const gn=100;
    const galleries: [string,(n:number)=>any][] = [
        ["hilbert",n=>Matrix.gallery.hilbert(n)],
        ["pascal (float)",n=>Matrix.gallery.pascal(n,"float")],
        ["pascal (exact)",n=>Matrix.gallery.pascal(n,"exact")],
        ["magic",n=>Matrix.gallery.magic(n)],
        ["lehmer",n=>Matrix.gallery.lehmer(n)],
        ["minij",n=>Matrix.gallery.minij(n)],
        ["fiedler",n=>Matrix.gallery.fiedler(n)],
        ["wilkinson",n=>Matrix.gallery.wilkinson(n)],
        ["grcar",n=>Matrix.gallery.grcar(n)],
        ["kahan",n=>Matrix.gallery.kahan(n)],
        ["invhess",n=>Matrix.gallery.invhess(n)],
        ["cauchy",n=>Matrix.gallery.cauchy(Array.from({length:n},(_,i)=>i+1),Array.from({length:n},(_,i)=>i+.5))],
        ["tridiag",n=>Matrix.gallery.tridiag(Array(n-1).fill(-1),Array(n).fill(4),Array(n-1).fill(-1))],
        ["binomial",n=>Matrix.gallery.binomial(n)],
    ];
    for (const [name,fn] of galleries) {
        const m=measure(()=>fn(gn));
        rec("Gallery",name,gn,"Float64M",m);
        console.log(`  ${name.padEnd(22)} ${m.avgMs.toFixed(3).padStart(8)}ms`);
    }
    const mW=measure(()=>Matrix.gallery.wathen(10,10));
    rec("Gallery","wathen 10×10",10,"Float64M",mW);
    console.log(`  ${"wathen 10×10".padEnd(22)} ${mW.avgMs.toFixed(3).padStart(8)}ms`);
}

// ── HTML REPORT ──────────────────────────────────────────────
function saveHTML() {
    const TC: Record<string,string> = {"Float64M":"#2196F3","Complex":"#E91E63","Rational":"#4CAF50"};

    // ── helper: genera dataset per una categoria + sizes ──
    function chartDatasets(cat: string, ops: string[], type: string, sizes: number[], palIn?: string[]) {
        const pal = palIn ?? ops.map((_,i) => `hsl(${i*40},65%,50%)`);
        return ops.map((op,ci) => {
            const pts = sizes.map(n => results.find(r=>r.category===cat&&r.operation===op&&r.size===n&&r.type===type)?.avgMs ?? null);
            return `{label:${JSON.stringify(op)},data:${JSON.stringify(pts)},borderColor:"${pal[ci]}",backgroundColor:"${pal[ci]}33",tension:.3,pointRadius:4,fill:false}`;
        }).join(",");
    }

    const SMALL_LABELS = JSON.stringify(CFG.small.map(n=>n+"×"+n));
    const MEDIUM_LABELS = JSON.stringify(CFG.medium.map(n=>n+"×"+n));
    const PAL_DECOMP = ["#2196F3","#4CAF50","#E91E63","#FF9800","#9C27B0","#795548","#00BCD4","#F44336"];
    const PAL_SOLVER = ["#1565C0","#0D47A1","#1976D2","#4CAF50","#388E3C","#F44336","#C62828","#E65100","#FF9800","#9C27B0"];
    const PAL_OPS    = ["#1565C0","#1976D2","#42A5F5","#64B5F6","#0097A7","#00838F"];
    const PAL_PROPS  = ["#1B5E20","#2E7D32","#388E3C","#43A047","#66BB6A","#81C784","#A5D6A7","#C8E6C9","#B71C1C","#C62828","#D32F2F","#E53935","#EF9A9A","#4A148C","#6A1B9A","#7B1FA2"];

    // ── tabelle HTML ──
    const solverRows = results.filter(r=>r.category==="Solver")
        .sort((a,b)=>a.size-b.size||a.operation.localeCompare(b.operation))
        .map(r=>{
            const bc=r.type==="Float64M"?"badge-f":r.type==="Complex"?"badge-c":"badge-r";
            const resStr=r.residual!==undefined
                ?`<span class="${r.residual<1e-8?"ok":r.residual<1e-4?"warn":"err"}">${r.residual.toExponential(2)}</span>`:"—";
            return `<tr><td>${r.operation}</td><td><span class="badge ${bc}">${r.type}</span></td><td>${r.size}</td><td>${r.avgMs}</td><td>${r.stdMs}</td><td>${resStr}</td></tr>`;
        }).join("\n");

    const propRows = results.filter(r=>r.category==="Proprietà")
        .sort((a,b)=>a.size-b.size||a.operation.localeCompare(b.operation))
        .map(r=>`<tr><td>${r.operation}</td><td>${r.size}</td><td>${r.avgMs}</td><td>${r.minMs}</td><td>${r.stdMs}</td></tr>`).join("\n");

    const allRows = [...results].sort((a,b)=>a.category.localeCompare(b.category)||a.size-b.size)
        .map(r=>{
            const bc=r.type==="Float64M"?"badge-f":r.type==="Complex"?"badge-c":"badge-r";
            return `<tr><td>${r.category}</td><td>${r.operation}</td><td><span class="badge ${bc}">${r.type}</span></td><td>${r.size}</td><td>${r.avgMs}</td><td>${r.stdMs}</td><td>${r.memDeltaMB}</td></tr>`;
        }).join("\n");

    // ── dati per grafici scalabilità per operazione ──
    const ARITM_OPS  = ["add","sub","mul","mul scalar","dotMul","dotDiv","dotPow(2)","pow(2)"];
    const UNARY_OPS  = ["abs","negate","round","sqrt","exp","sin","cos"];
    const TRANSF_OPS = ["transpose","ct","flip ud","flip lr","rot90"];
    const STAT_OPS   = ["sum col","sum row","mean","max","totalSum","norm Fro","norm 1","norm Inf","trace"];
    const PROP_OPS   = ["isSquare","isSymmetric","isUpperTriangular","isLowerTriangular","isDiagonal","isIdentity","isOrthogonal","isZeroMatrix","isInvertible","isSingular","isPositiveDefinite","isPositiveSemiDef","isDiagonallyDominant","hasZeroTrace","hasFiniteValues","isStochastic"];
    const DECOMP_OPS = ["LUP","LU","LU total","QR","Cholesky","LDLT"];
    const SOLVER_OPS = ["LUP","LU","QR","Cholesky","LDLT","Jacobi","Gauss-Seidel","SOR ω=1.0","SOR ω=1.5","CG (SPD)"];

    const html = `<!DOCTYPE html><html lang="it"><head><meta charset="UTF-8">
<title>numeric-matrix — Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#f0f2f5;color:#222;font-size:14px}
header{background:#1565C0;color:#fff;padding:18px 36px}
header h1{font-size:1.5rem;font-weight:600}
header p{opacity:.75;font-size:.82rem;margin-top:4px}
main{max-width:1600px;margin:0 auto;padding:18px}
nav{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}
nav a{background:#E3F2FD;color:#1565C0;padding:5px 12px;border-radius:16px;text-decoration:none;font-size:.8rem;font-weight:500}
nav a:hover{background:#BBDEFB}
section{background:#fff;border-radius:8px;padding:16px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
h2{font-size:.95rem;font-weight:600;color:#1565C0;margin-bottom:12px;border-bottom:2px solid #E3F2FD;padding-bottom:6px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.chart-wrap{position:relative;height:300px}
.chart-wrap-tall{position:relative;height:380px}
table{width:100%;border-collapse:collapse;font-size:.76rem}
th{background:#E3F2FD;color:#0D47A1;padding:5px 8px;text-align:left;border-bottom:2px solid #90CAF9;white-space:nowrap}
td{padding:3px 8px;border-bottom:1px solid #eee}
tr:hover td{background:#F8FBFF}
.ok{color:#2E7D32;font-weight:600}.warn{color:#E65100;font-weight:600}.err{color:#C62828;font-weight:600}
.badge{display:inline-block;padding:1px 6px;border-radius:10px;font-size:.7rem;font-weight:600}
.badge-f{background:#BBDEFB;color:#0D47A1}.badge-c{background:#FCE4EC;color:#880E4F}.badge-r{background:#E8F5E9;color:#1B5E20}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:16px}
.card{background:#E3F2FD;border-radius:6px;padding:10px 14px}
.card .val{font-size:1.4rem;font-weight:700;color:#0D47A1}
.card .lbl{font-size:.68rem;color:#546E7A;margin-top:2px}
.note{background:#FFF3E0;border-left:3px solid #FF9800;padding:8px 12px;margin-bottom:12px;font-size:.8rem;border-radius:3px}
</style></head><body>
<header>
  <h1>numeric-matrix — Benchmark Report Completo</h1>
  <p>Float64M · Complex · Rational · ${new Date().toLocaleString("it-IT")} · Node ${process.version}</p>
</header>
<main>

<nav>
  <a href="#summary">Riepilogo</a>
  <a href="#scalabilita">Scalabilità mul</a>
  <a href="#aritmetica">Aritmetica</a>
  <a href="#unarie">Unarie</a>
  <a href="#trasformazioni">Trasformazioni</a>
  <a href="#statistiche">Statistiche</a>
  <a href="#decomposizioni">Decomposizioni</a>
  <a href="#solver">Solver</a>
  <a href="#iterativi">Iterativi</a>
  <a href="#proprieta">Proprietà</a>
  <a href="#tipi">Confronto tipi</a>
  <a href="#tabelle">Tabelle</a>
</nav>

<section id="summary">
  <h2>Riepilogo</h2>
  <div class="summary">
    <div class="card"><div class="val">${results.length}</div><div class="lbl">Misurazioni</div></div>
    <div class="card"><div class="val">${[...new Set(results.map(r=>r.category))].length}</div><div class="lbl">Categorie</div></div>
    <div class="card"><div class="val">${[...new Set(results.map(r=>r.type))].length}</div><div class="lbl">Tipi numerici</div></div>
    <div class="card"><div class="val">${results.filter(r=>r.residual!==undefined&&r.residual<1e-8).length}</div><div class="lbl">Solver ✓ res&lt;1e-8</div></div>
    <div class="card"><div class="val">${results.filter(r=>r.residual!==undefined&&(r.residual>=1e-4||isNaN(r.residual??NaN))).length}</div><div class="lbl">Solver con problemi</div></div>
    <div class="card"><div class="val">${[...new Set(results.map(r=>r.operation))].length}</div><div class="lbl">Operazioni distinte</div></div>
  </div>
</section>

<section id="scalabilita">
  <h2>Scalabilità — Moltiplicazione matriciale vs O(n³)</h2>
  <div class="chart-wrap-tall"><canvas id="c_mul_scale"></canvas></div>
</section>

<div class="grid2" id="aritmetica">
<section>
  <h2>Aritmetica element-wise — scalabilità</h2>
  <div class="chart-wrap"><canvas id="c_aritm"></canvas></div>
</section>
<section id="unarie">
  <h2>Funzioni unarie — scalabilità</h2>
  <div class="chart-wrap"><canvas id="c_unary"></canvas></div>
</section>
</div>

<div class="grid2" id="trasformazioni">
<section>
  <h2>Trasformazioni — scalabilità</h2>
  <div class="chart-wrap"><canvas id="c_transf"></canvas></div>
</section>
<section id="statistiche">
  <h2>Statistiche e norme — scalabilità</h2>
  <div class="chart-wrap"><canvas id="c_stat"></canvas></div>
</section>
</div>

<section id="decomposizioni">
  <h2>Decomposizioni — confronto metodi</h2>
  <div class="chart-wrap-tall"><canvas id="c_decomp"></canvas></div>
</section>

<section id="solver">
  <h2>Solver — tempo per dimensione</h2>
  <div class="note">Tutti i solver usano matrici appropriate: Cholesky/LDLT/CG su SPD, gli iterativi su matrice diagonalmente dominante. Il residuo indica la qualità numerica.</div>
  <div class="grid2">
    <div><div class="chart-wrap"><canvas id="c_solver_time"></canvas></div></div>
    <div><div class="chart-wrap"><canvas id="c_solver_res"></canvas></div></div>
  </div>
</section>

<section id="iterativi">
  <h2>Metodi iterativi — analisi</h2>
  <div class="grid2">
    <div>
      <h2 style="font-size:.85rem;color:#555">SOR: residuo e tempo al variare di ω (n=50, matrice SPD+diag.dom.)</h2>
      <div class="chart-wrap"><canvas id="c_sor"></canvas></div>
    </div>
    <div>
      <h2 style="font-size:.85rem;color:#555">Confronto metodi su stessa matrice SPD+diag.dom. (n=50)</h2>
      <div class="chart-wrap"><canvas id="c_iter_cmp"></canvas></div>
    </div>
  </div>
</section>

<section id="proprieta">
  <h2>Proprietà strutturali — tutte (scalabilità)</h2>
  <div class="note">Ogni proprietà usa la matrice più adatta (es. isSymmetric su matrice simmetrica, isZeroMatrix su matrice zero). isInvertible e isPositiveDefinite sono O(n³) (fanno LUP/Cholesky internamente).</div>
  <div class="grid2">
    <div><div class="chart-wrap-tall"><canvas id="c_prop_fast"></canvas></div></div>
    <div><div class="chart-wrap-tall"><canvas id="c_prop_slow"></canvas></div></div>
  </div>
</section>

<section id="tipi">
  <h2>Confronto tipi numerici — mul e LUP</h2>
  <div class="grid2">
    <div><div class="chart-wrap"><canvas id="c_types_mul"></canvas></div></div>
    <div><div class="chart-wrap"><canvas id="c_complex_solver"></canvas></div></div>
  </div>
</section>

<section id="tabelle">
  <h2>Solver — dettaglio residui</h2>
  <table><thead><tr><th>Metodo</th><th>Tipo</th><th>n</th><th>Avg ms</th><th>σ ms</th><th>Residuo</th></tr></thead>
  <tbody>${solverRows}</tbody></table>
</section>

<section>
  <h2>Proprietà strutturali — dettaglio tempi</h2>
  <table><thead><tr><th>Proprietà</th><th>n</th><th>Avg ms</th><th>Min ms</th><th>σ ms</th></tr></thead>
  <tbody>${propRows}</tbody></table>
</section>

<section>
  <h2>Tutte le misurazioni</h2>
  <table><thead><tr><th>Categoria</th><th>Operazione</th><th>Tipo</th><th>n</th><th>Avg ms</th><th>σ ms</th><th>Mem MB</th></tr></thead>
  <tbody>${allRows}</tbody></table>
</section>

</main>
<script>
const R=${JSON.stringify(results)};
function get(cat,op,type,size){return R.find(r=>r.category===cat&&r.operation===op&&r.type===type&&r.size===size)?.avgMs??null;}
function series(cat,ops,type,sizes,pals){
  return ops.map((op,ci)=>({label:op,data:sizes.map(n=>get(cat,op,type,n)),borderColor:pals[ci]??'#888',backgroundColor:(pals[ci]??'#888')+'33',tension:.3,pointRadius:4,fill:false}));
}
function chart(id,labels,datasets,yLabel,log=false){
  new Chart(document.getElementById(id),{type:"line",data:{labels,datasets},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{position:"bottom",labels:{boxWidth:12,font:{size:11}}}},
      scales:{y:{type:log?"logarithmic":"linear",title:{display:true,text:yLabel},beginAtZero:!log},
              x:{title:{display:true,text:"n"}}}}});
}

const SM=${SMALL_LABELS}, MED=${MEDIUM_LABELS};
const PM=${JSON.stringify([50,100,200,400].map(n=>n+"×"+n))};

// 1. Scalabilità mul
(()=>{
  const rows=R.filter(r=>r.category==="Scalabilità"&&r.operation==="mul A×B").sort((a,b)=>a.size-b.size);
  const labels=rows.map(r=>r.size+"×"+r.size);
  const theory=rows.map(r=>rows[0].avgMs*(r.size/rows[0].size)**3);
  chart("c_mul_scale",labels,[
    {label:"Float64M measured",data:rows.map(r=>r.avgMs),borderColor:"#2196F3",backgroundColor:"#2196F322",fill:true,tension:.3,pointRadius:5},
    {label:"O(n³) teorico",data:theory,borderColor:"#999",borderDash:[5,4],fill:false,pointRadius:0}
  ],"ms");
})();

// 2. Aritmetica
chart("c_aritm",MED,series("Aritmetica",["add","sub","mul","dotMul","dotDiv"],"Float64M",${JSON.stringify(CFG.medium)},["#1565C0","#1976D2","#E53935","#1E88E5","#42A5F5"]),"ms",true);

// 3. Unarie
chart("c_unary",MED,series("Unarie",["abs","negate","sqrt","round","sin","cos"],"Float64M",${JSON.stringify(CFG.medium)},["#B71C1C","#C62828","#D32F2F","#E53935","#EF5350","#FF8A80"]),"ms");

// 4. Trasformazioni
chart("c_transf",MED,series("Trasform.",["transpose","ct","flip ud","flip lr","rot90"],"Float64M",${JSON.stringify(CFG.medium)},["#1B5E20","#2E7D32","#388E3C","#43A047","#66BB6A"]),"ms");

// 5. Statistiche
chart("c_stat",MED,series("Stat.",["sum col","sum row","mean","norm Fro","norm 1","norm Inf","totalSum","trace"],"Float64M",${JSON.stringify(CFG.medium)},["#E65100","#EF6C00","#F57C00","#FB8C00","#FFA726","#FFB74D","#FFCC80","#FF8A65"]),"ms");

// 6. Decomposizioni
chart("c_decomp",SM,series("Decomp.",["LUP","LU","LU total","QR","Cholesky","LDLT"],"Float64M",${JSON.stringify(CFG.small)},${JSON.stringify(PAL_DECOMP.slice(0,6))}),"ms",true);

// 7. Solver tempo
chart("c_solver_time",SM,series("Solver",["LUP","LU","QR","Cholesky","LDLT","Jacobi","Gauss-Seidel","SOR ω=1.0","SOR ω=1.5","CG (SPD)"],"Float64M",${JSON.stringify(CFG.small)},${JSON.stringify(PAL_SOLVER)}),"ms",true);

// 8. Solver residuo (scatter)
(()=>{
  const solvers=["LUP","LU","QR","Cholesky","LDLT","Jacobi","Gauss-Seidel","SOR ω=1.0","SOR ω=1.5","CG (SPD)"];
  const pal=${JSON.stringify(PAL_SOLVER)};
  const datasets=solvers.map((op,ci)=>({
    label:op,type:"scatter",
    data:R.filter(r=>r.category==="Solver"&&r.operation===op&&r.type==="Float64M"&&r.residual!==undefined)
          .map(r=>({x:r.size,y:r.residual})),
    backgroundColor:pal[ci]??"#888",pointRadius:7,
  }));
  new Chart(document.getElementById("c_solver_res"),{type:"scatter",data:{datasets},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{position:"bottom",labels:{boxWidth:12,font:{size:11}}}},
      scales:{x:{title:{display:true,text:"n"}},y:{type:"logarithmic",title:{display:true,text:"Residuo relativo"}}}}});
})();

// 9. SOR omega
(()=>{
  const rows=R.filter(r=>r.category==="Iterativi"&&r.operation.startsWith("SOR ω=")&&r.size===50).sort((a,b)=>parseFloat(a.operation.split("=")[1])-parseFloat(b.operation.split("=")[1]));
  const labels=rows.map(r=>r.operation.split("=")[1]);
  new Chart(document.getElementById("c_sor"),{type:"line",data:{labels,datasets:[
    {label:"Residuo",data:rows.map(r=>r.residual??null),borderColor:"#E91E63",yAxisID:"y1",pointRadius:7,fill:false,tension:.2},
    {label:"Tempo ms",data:rows.map(r=>r.avgMs),borderColor:"#2196F3",yAxisID:"y2",borderDash:[4,3],pointRadius:7,fill:false,tension:.2},
  ]},options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{position:"bottom"}},
    scales:{y1:{type:"logarithmic",title:{display:true,text:"Residuo"},position:"left"},
            y2:{title:{display:true,text:"ms"},position:"right",grid:{drawOnChartArea:false}}}}});
})();

// 10. Confronto iterativi vs diretti su stessa matrice
(()=>{
  const ops=["Jacobi","Gauss-Seidel","SOR ω=1.0","SOR ω=1.5","CG","Cholesky","LUP"];
  const row=R.filter(r=>r.category==="Iterativi vs Diretti"&&r.size===50);
  const pal=["#F44336","#FF9800","#E65100","#FFC107","#4CAF50","#2196F3","#1565C0"];
  const labels=ops.map(o=>{const r2=row.find(r=>r.operation===o);return r2?o+" ("+r2.avgMs+"ms)":o;});
  new Chart(document.getElementById("c_iter_cmp"),{type:"bar",data:{
    labels,
    datasets:[
      {label:"Tempo ms",data:ops.map(o=>row.find(r=>r.operation===o)?.avgMs??null),backgroundColor:pal.map(c=>c+"CC")},
    ]},options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{y:{title:{display:true,text:"ms"},type:"logarithmic"},x:{ticks:{font:{size:10}}}}}});
})();

// 11. Proprietà: separa fast (O(n²)) da slow (O(n³))
(()=>{
  const fast=["isSquare","isSymmetric","isUpperTriangular","isLowerTriangular","isDiagonal","isZeroMatrix","isDiagonallyDominant","hasZeroTrace","hasFiniteValues","isStochastic","isPositiveSemiDef"];
  const slow=["isIdentity","isOrthogonal","isInvertible","isSingular","isPositiveDefinite"];
  const palF=["#1B5E20","#2E7D32","#388E3C","#43A047","#66BB6A","#81C784","#A5D6A7","#E8F5E9","#558B2F","#33691E","#827717"];
  const palS=["#B71C1C","#C62828","#D32F2F","#E53935","#EF5350"];
  chart("c_prop_fast",PM,series("Proprietà",fast,"Float64M",[50,100,200,400],palF),"ms");
  chart("c_prop_slow",PM,series("Proprietà",slow,"Float64M",[50,100,200,400],palS),"ms",true);
})();

// 12. Confronto tipi mul
(()=>{
  const sizes=[5,10,20,30];
  const labels=sizes.map(n=>n+"×"+n);
  new Chart(document.getElementById("c_types_mul"),{type:"bar",data:{labels,datasets:[
    {label:"Float64M",data:sizes.map(n=>get("Confronto","mul F64","Float64M",n)),backgroundColor:"#2196F3CC"},
    {label:"Complex", data:sizes.map(n=>get("Confronto","mul Cmx","Complex",n)),backgroundColor:"#E91E63CC"},
    {label:"Rational",data:sizes.map(n=>get("Confronto","mul Rat","Rational",n)),backgroundColor:"#4CAF50CC"},
  ]},options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{position:"bottom"}},
    scales:{y:{type:"logarithmic",title:{display:true,text:"ms (log)"}},x:{title:{display:true,text:"n"}}}}});
})();

// 13. Complex solver
chart("c_complex_solver",[5,10,25,50,100].map(n=>n+"×"+n),
  series("Solver",["LUP (C)","QR (C)"],"Complex",[5,10,25,50,100],["#E91E63","#880E4F"]),"ms",true);

</script></body></html>`;

    writeFileSync("benchmark/benchmark_report.html", html);
    console.log("  ✓ benchmark/benchmark_report.html");
}

// ── MAIN ─────────────────────────────────────────────────────
async function main() {
    console.log("╔══════════════════════════════════════════════════════╗");
    console.log("║        numeric-matrix — BENCHMARK COMPLETO          ║");
    console.log(`║  Node ${process.version.padEnd(10)} · ${new Date().toLocaleString("it-IT").padEnd(26)}║`);
    console.log("╚══════════════════════════════════════════════════════╝");

    benchFloat64();
    benchDecompositions();
    benchSolvers();
    benchMulScaling();
    benchIterativeConvergence();
    benchProperties();
    benchComplex();
    benchRational();
    benchTypeComparison();
    benchGallery();

    console.log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    const r3=(x:number,d=3)=>Math.round(x*10**d)/10**d;
    writeFileSync("benchmark/benchmark_results.json",
        JSON.stringify({meta:{date:new Date().toISOString(),node:process.version},results},null,2));
    console.log("  ✓ benchmark/benchmark_results.json");
    saveHTML();
    console.log("\n  Apri benchmark/benchmark_report.html nel browser.");
    console.log("  Per benchmark grandi: npx tsx --expose-gc benchmark/benchmark_simple.ts");
}

main().catch(console.error);

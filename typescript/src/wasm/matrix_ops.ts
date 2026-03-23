// =============================================================
// src/wasm/matrix_ops.ts  —  AssemblyScript source
//
// Compile:
//   asc src/wasm/matrix_ops.ts --outFile src/wasm/matrix_ops.wasm \
//     --optimizeLevel 3 --shrinkLevel 0 --runtime stub --noAssert \
//     --enable simd
//
// Ottimizzazioni chiave:
//   • SIMD f64x2: ogni ciclo processa 2 double → ~2× throughput su op elementali
//   • Matmul a blocchi (TILE 64×64): riuso cache L1/L2, loop i-k-j con SIMD sul j
//   • Matvec SIMD: dot-product con f64x2 accumulator
//   • Solver iterativi completi in WASM (Jacobi, GS, SOR, CG, JOR):
//     zero boundary-crossing JS per iterazione, zero GC pressure
//   • Trasposta cache-blocked (tile 32×32)
//   • Property checks inline senza call overhead
//   • diagInv precomputed per Jacobi/JOR: sostituisce n div/iter con n mul
//   • Convergenza relativa: ||dx||/max(||x||,1) < tol (piu robusta dell'assoluta)
//   • GS/SOR loop separati j<i e j>i: zero branch nel loop SIMD
// =============================================================

// ─── helpers scalari ─────────────────────────────────────────────────────────
@inline function ld(ptr: i32, i: i32): f64  { return load<f64>(ptr + (i << 3)); }
@inline function st(ptr: i32, i: i32, v: f64): void { store<f64>(ptr + (i << 3), v); }
@inline function fAbs(v: f64): f64  { return v < 0.0 ? -v : v; }
@inline function fMin(a: f64, b: f64): f64  { return a < b ? a : b; }
@inline function fMax(a: f64, b: f64): f64  { return a > b ? a : b; }
@inline function iMin(a: i32, b: i32): i32  { return a < b ? a : b; }

// ─── helpers SIMD ─────────────────────────────────────────────────────────────
@inline function ld2(ptr: i32, byteOff: i32): v128 { return v128.load(ptr + byteOff); }
@inline function st2(ptr: i32, byteOff: i32, v: v128): void { v128.store(ptr + byteOff, v); }
@inline function splat2(v: f64): v128 { return f64x2.splat(v); }

@inline function _dotN(aOff: i32, bOff: i32, n: i32): f64 {
    let vs: v128 = f64x2.splat(0.0);
    let i: i32 = 0;
    for (; i + 1 < n; i += 2) {
        const off = i << 3;
        vs = f64x2.add(vs, f64x2.mul(v128.load(aOff + off), v128.load(bOff + off)));
    }
    let s: f64 = f64x2.extract_lane(vs, 0) + f64x2.extract_lane(vs, 1);
    if (i < n) s += ld(aOff, i) * ld(bOff, i);
    return s;
}

export function zeroF64(ptr: i32, len: i32): void {
    const z = f64x2.splat(0.0);
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) v128.store(ptr + (i << 3), z);
    if (i < len) store<f64>(ptr + (i << 3), 0.0);
}

// ─── 1. ELEMENT-WISE (SIMD) ───────────────────────────────────────────────────

export function addMatrix(aOff: i32, bOff: i32, cOff: i32, len: i32): void {
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) {
        const off = i << 3;
        st2(cOff, off, f64x2.add(ld2(aOff, off), ld2(bOff, off)));
    }
    if (i < len) st(cOff, i, ld(aOff, i) + ld(bOff, i));
}

export function subMatrix(aOff: i32, bOff: i32, cOff: i32, len: i32): void {
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) {
        const off = i << 3;
        st2(cOff, off, f64x2.sub(ld2(aOff, off), ld2(bOff, off)));
    }
    if (i < len) st(cOff, i, ld(aOff, i) - ld(bOff, i));
}

export function dotMul(aOff: i32, bOff: i32, cOff: i32, len: i32): void {
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) {
        const off = i << 3;
        st2(cOff, off, f64x2.mul(ld2(aOff, off), ld2(bOff, off)));
    }
    if (i < len) st(cOff, i, ld(aOff, i) * ld(bOff, i));
}

export function dotDiv(aOff: i32, bOff: i32, cOff: i32, len: i32): void {
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) {
        const off = i << 3;
        st2(cOff, off, f64x2.div(ld2(aOff, off), ld2(bOff, off)));
    }
    if (i < len) st(cOff, i, ld(aOff, i) / ld(bOff, i));
}

export function addScalar(aOff: i32, cOff: i32, len: i32, scalar: f64): void {
    const vs = splat2(scalar);
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) {
        const off = i << 3;
        st2(cOff, off, f64x2.add(ld2(aOff, off), vs));
    }
    if (i < len) st(cOff, i, ld(aOff, i) + scalar);
}

export function subScalar(aOff: i32, cOff: i32, len: i32, scalar: f64): void {
    const vs = splat2(scalar);
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) {
        const off = i << 3;
        st2(cOff, off, f64x2.sub(ld2(aOff, off), vs));
    }
    if (i < len) st(cOff, i, ld(aOff, i) - scalar);
}

export function mulScalar(aOff: i32, cOff: i32, len: i32, scalar: f64): void {
    const vs = splat2(scalar);
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) {
        const off = i << 3;
        st2(cOff, off, f64x2.mul(ld2(aOff, off), vs));
    }
    if (i < len) st(cOff, i, ld(aOff, i) * scalar);
}

export function dotPowScalar(aOff: i32, cOff: i32, len: i32, exp: f64): void {
    for (let i: i32 = 0; i < len; i++) st(cOff, i, Math.pow(ld(aOff, i), exp));
}

// ─── 2. BROADCAST (SIMD) ─────────────────────────────────────────────────────

export function addRowVec(aOff: i32, bOff: i32, cOff: i32, R: i32, C: i32): void {
    for (let i: i32 = 0; i < R; i++) {
        const rowByteOff = i * C * 8;
        let j: i32 = 0;
        for (; j + 1 < C; j += 2) {
            const off = j << 3;
            st2(cOff + rowByteOff, off, f64x2.add(ld2(aOff + rowByteOff, off), ld2(bOff, off)));
        }
        if (j < C) {
            const rOff = i * C + j;
            st(cOff, rOff, ld(aOff, rOff) + ld(bOff, j));
        }
    }
}

export function subRowVec(aOff: i32, bOff: i32, cOff: i32, R: i32, C: i32): void {
    for (let i: i32 = 0; i < R; i++) {
        const rowByteOff = i * C * 8;
        let j: i32 = 0;
        for (; j + 1 < C; j += 2) {
            const off = j << 3;
            st2(cOff + rowByteOff, off, f64x2.sub(ld2(aOff + rowByteOff, off), ld2(bOff, off)));
        }
        if (j < C) {
            const rOff = i * C + j;
            st(cOff, rOff, ld(aOff, rOff) - ld(bOff, j));
        }
    }
}

export function addColVec(aOff: i32, bOff: i32, cOff: i32, R: i32, C: i32): void {
    for (let i: i32 = 0; i < R; i++) {
        const vs = splat2(ld(bOff, i));
        const rowByteOff = i * C * 8;
        let j: i32 = 0;
        for (; j + 1 < C; j += 2) {
            const off = j << 3;
            st2(cOff + rowByteOff, off, f64x2.add(ld2(aOff + rowByteOff, off), vs));
        }
        if (j < C) {
            const rOff = i * C + j;
            st(cOff, rOff, ld(aOff, rOff) + ld(bOff, i));
        }
    }
}

export function subColVec(aOff: i32, bOff: i32, cOff: i32, R: i32, C: i32): void {
    for (let i: i32 = 0; i < R; i++) {
        const vs = splat2(ld(bOff, i));
        const rowByteOff = i * C * 8;
        let j: i32 = 0;
        for (; j + 1 < C; j += 2) {
            const off = j << 3;
            st2(cOff + rowByteOff, off, f64x2.sub(ld2(aOff + rowByteOff, off), vs));
        }
        if (j < C) {
            const rOff = i * C + j;
            st(cOff, rOff, ld(aOff, rOff) - ld(bOff, i));
        }
    }
}

export function dotMulRowVec(aOff: i32, bOff: i32, cOff: i32, R: i32, C: i32): void {
    for (let i: i32 = 0; i < R; i++) {
        const rowByteOff = i * C * 8;
        let j: i32 = 0;
        for (; j + 1 < C; j += 2) {
            const off = j << 3;
            st2(cOff + rowByteOff, off, f64x2.mul(ld2(aOff + rowByteOff, off), ld2(bOff, off)));
        }
        if (j < C) {
            const rOff = i * C + j;
            st(cOff, rOff, ld(aOff, rOff) * ld(bOff, j));
        }
    }
}

export function dotMulColVec(aOff: i32, bOff: i32, cOff: i32, R: i32, C: i32): void {
    for (let i: i32 = 0; i < R; i++) {
        const vs = splat2(ld(bOff, i));
        const rowByteOff = i * C * 8;
        let j: i32 = 0;
        for (; j + 1 < C; j += 2) {
            const off = j << 3;
            st2(cOff + rowByteOff, off, f64x2.mul(ld2(aOff + rowByteOff, off), vs));
        }
        if (j < C) {
            const rOff = i * C + j;
            st(cOff, rOff, ld(aOff, rOff) * ld(bOff, i));
        }
    }
}

// ─── 3. MATMUL A BLOCCHI + SIMD ──────────────────────────────────────────────

const TILE: i32 = 64;

export function matmul(aOff: i32, bOff: i32, cOff: i32, M: i32, K: i32, N: i32): void {
    zeroF64(cOff, M * N);
    for (let ii: i32 = 0; ii < M; ii += TILE) {
        const iEnd: i32 = iMin(ii + TILE, M);
        for (let kk: i32 = 0; kk < K; kk += TILE) {
            const kEnd: i32 = iMin(kk + TILE, K);
            for (let jj: i32 = 0; jj < N; jj += TILE) {
                const jEnd: i32 = iMin(jj + TILE, N);
                for (let i: i32 = ii; i < iEnd; i++) {
                    const iK: i32 = i * K, iN: i32 = i * N;
                    const cRow: i32 = cOff + iN * 8;
                    for (let k: i32 = kk; k < kEnd; k++) {
                        const aik: f64 = ld(aOff, iK + k);
                        const vaik: v128 = splat2(aik);
                        const bRow: i32 = bOff + k * N * 8;
                        let j: i32 = jj;
                        for (; j + 1 < jEnd; j += 2) {
                            const jByteOff: i32 = j * 8;
                            st2(cRow, jByteOff,
                                f64x2.add(v128.load(cRow + jByteOff),
                                          f64x2.mul(vaik, v128.load(bRow + jByteOff))));
                        }
                        if (j < jEnd) {
                            const jByteOff: i32 = j * 8;
                            store<f64>(cRow + jByteOff,
                                load<f64>(cRow + jByteOff) + aik * load<f64>(bRow + jByteOff));
                        }
                    }
                }
            }
        }
    }
}

// ─── 4. MATVEC (SIMD) ────────────────────────────────────────────────────────

export function matvec(aOff: i32, xOff: i32, yOff: i32, n: i32): void {
    for (let i: i32 = 0; i < n; i++) {
        st(yOff, i, _dotN(aOff + i * n * 8, xOff, n));
    }
}

// ─── 5. TRASPOSTA CACHE-BLOCKED ───────────────────────────────────────────────

const TTILE: i32 = 32;

export function transpose(aOff: i32, cOff: i32, R: i32, C: i32): void {
    for (let ii: i32 = 0; ii < R; ii += TTILE) {
        const iEnd: i32 = iMin(ii + TTILE, R);
        for (let jj: i32 = 0; jj < C; jj += TTILE) {
            const jEnd: i32 = iMin(jj + TTILE, C);
            for (let i: i32 = ii; i < iEnd; i++)
                for (let j: i32 = jj; j < jEnd; j++)
                    store<f64>(cOff + (j * R + i) * 8, load<f64>(aOff + (i * C + j) * 8));
        }
    }
}

// ─── 6. NORME (SIMD) ─────────────────────────────────────────────────────────

export function normFro(aOff: i32, len: i32): f64 {
    let vs: v128 = f64x2.splat(0.0);
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) {
        const v: v128 = ld2(aOff, i << 3);
        vs = f64x2.add(vs, f64x2.mul(v, v));
    }
    let ss: f64 = f64x2.extract_lane(vs, 0) + f64x2.extract_lane(vs, 1);
    if (i < len) { const v = ld(aOff, i); ss += v * v; }
    return Math.sqrt(ss);
}

export function normVec1(aOff: i32, len: i32): f64 {
    let s: f64 = 0.0;
    for (let i: i32 = 0; i < len; i++) { const v = ld(aOff, i); s += v < 0.0 ? -v : v; }
    return s;
}

export function normVecInf(aOff: i32, len: i32): f64 {
    let m: f64 = 0.0;
    for (let i: i32 = 0; i < len; i++) { const v = fAbs(ld(aOff, i)); if (v > m) m = v; }
    return m;
}

export function normMat1(aOff: i32, R: i32, C: i32): f64 {
    let maxS: f64 = 0.0;
    for (let j: i32 = 0; j < C; j++) {
        let s: f64 = 0.0;
        for (let i: i32 = 0; i < R; i++) { const v = ld(aOff, i * C + j); s += v < 0.0 ? -v : v; }
        if (s > maxS) maxS = s;
    }
    return maxS;
}

export function normMatInf(aOff: i32, R: i32, C: i32): f64 {
    let maxS: f64 = 0.0;
    for (let i: i32 = 0; i < R; i++) {
        const rowOff: i32 = aOff + i * C * 8;
        let vs: v128 = f64x2.splat(0.0);
        let j: i32 = 0;
        for (; j + 1 < C; j += 2) {
            const off = j << 3;
            vs = f64x2.add(vs, f64x2.abs(ld2(rowOff, off)));
        }
        let s: f64 = f64x2.extract_lane(vs, 0) + f64x2.extract_lane(vs, 1);
        if (j < C) s += fAbs(ld(aOff, i * C + j));
        if (s > maxS) maxS = s;
    }
    return maxS;
}

// ─── 7. FUNZIONI UNARIE (SIMD dove possibile) ─────────────────────────────────

export function unaryAbs(aOff: i32, cOff: i32, len: i32): void {
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) {
        const off = i << 3;
        st2(cOff, off, f64x2.abs(ld2(aOff, off)));
    }
    if (i < len) st(cOff, i, fAbs(ld(aOff, i)));
}

export function unaryNeg(aOff: i32, cOff: i32, len: i32): void {
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) {
        const off = i << 3;
        st2(cOff, off, f64x2.neg(ld2(aOff, off)));
    }
    if (i < len) st(cOff, i, -ld(aOff, i));
}

export function unarySqrt(aOff: i32, cOff: i32, len: i32): void {
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) {
        const off = i << 3;
        st2(cOff, off, f64x2.sqrt(ld2(aOff, off)));
    }
    if (i < len) st(cOff, i, Math.sqrt(ld(aOff, i)));
}

export function unaryRound(aOff: i32, cOff: i32, len: i32): void {
    for (let i: i32 = 0; i < len; i++) st(cOff, i, Math.round(ld(aOff, i)));
}

export function unaryFloor(aOff: i32, cOff: i32, len: i32): void {
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) {
        const off = i << 3;
        st2(cOff, off, f64x2.floor(ld2(aOff, off)));
    }
    if (i < len) st(cOff, i, Math.floor(ld(aOff, i)));
}

export function unaryCeil(aOff: i32, cOff: i32, len: i32): void {
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) {
        const off = i << 3;
        st2(cOff, off, f64x2.ceil(ld2(aOff, off)));
    }
    if (i < len) st(cOff, i, Math.ceil(ld(aOff, i)));
}

export function unaryExp(aOff: i32, cOff: i32, len: i32): void {
    for (let i: i32 = 0; i < len; i++) st(cOff, i, Math.exp(ld(aOff, i)));
}
export function unarySin(aOff: i32, cOff: i32, len: i32): void {
    for (let i: i32 = 0; i < len; i++) st(cOff, i, Math.sin(ld(aOff, i)));
}
export function unaryCos(aOff: i32, cOff: i32, len: i32): void {
    for (let i: i32 = 0; i < len; i++) st(cOff, i, Math.cos(ld(aOff, i)));
}
export function unaryTan(aOff: i32, cOff: i32, len: i32): void {
    for (let i: i32 = 0; i < len; i++) st(cOff, i, Math.tan(ld(aOff, i)));
}

// ─── 8. STATISTICHE (SIMD) ───────────────────────────────────────────────────

export function totalSum(aOff: i32, len: i32): f64 {
    let vs: v128 = f64x2.splat(0.0);
    let i: i32 = 0;
    for (; i + 1 < len; i += 2) vs = f64x2.add(vs, ld2(aOff, i << 3));
    let s: f64 = f64x2.extract_lane(vs, 0) + f64x2.extract_lane(vs, 1);
    if (i < len) s += ld(aOff, i);
    return s;
}

export function trace(aOff: i32, n: i32): f64 {
    let s: f64 = 0.0;
    for (let i: i32 = 0; i < n; i++) s += ld(aOff, i * n + i);
    return s;
}

export function sumCols(aOff: i32, outOff: i32, R: i32, C: i32): void {
    zeroF64(outOff, C);
    for (let i: i32 = 0; i < R; i++) {
        const rowOff: i32 = aOff + i * C * 8;
        let j: i32 = 0;
        for (; j + 1 < C; j += 2) {
            const off = j << 3;
            st2(outOff, off, f64x2.add(ld2(outOff, off), ld2(rowOff, off)));
        }
        if (j < C) st(outOff, j, ld(outOff, j) + ld(aOff, i * C + j));
    }
}

export function sumRows(aOff: i32, outOff: i32, R: i32, C: i32): void {
    for (let i: i32 = 0; i < R; i++) {
        let vs: v128 = f64x2.splat(0.0);
        let j: i32 = 0;
        for (; j + 1 < C; j += 2) vs = f64x2.add(vs, ld2(aOff + i * C * 8, j << 3));
        let s: f64 = f64x2.extract_lane(vs, 0) + f64x2.extract_lane(vs, 1);
        if (j < C) s += ld(aOff, i * C + j);
        st(outOff, i, s);
    }
}

export function maxCols(aOff: i32, outOff: i32, idxOff: i32, R: i32, C: i32): void {
    for (let j: i32 = 0; j < C; j++) {
        let best: f64 = ld(aOff, j); let bestI: i32 = 0;
        for (let i: i32 = 1; i < R; i++) {
            const v: f64 = ld(aOff, i * C + j);
            if (v > best) { best = v; bestI = i; }
        }
        st(outOff, j, best); store<i32>(idxOff + j * 4, bestI + 1);
    }
}

export function minCols(aOff: i32, outOff: i32, idxOff: i32, R: i32, C: i32): void {
    for (let j: i32 = 0; j < C; j++) {
        let best: f64 = ld(aOff, j); let bestI: i32 = 0;
        for (let i: i32 = 1; i < R; i++) {
            const v: f64 = ld(aOff, i * C + j);
            if (v < best) { best = v; bestI = i; }
        }
        st(outOff, j, best); store<i32>(idxOff + j * 4, bestI + 1);
    }
}

export function maxRows(aOff: i32, outOff: i32, idxOff: i32, R: i32, C: i32): void {
    for (let i: i32 = 0; i < R; i++) {
        const off: i32 = i * C;
        let best: f64 = ld(aOff, off); let bestJ: i32 = 0;
        for (let j: i32 = 1; j < C; j++) {
            const v: f64 = ld(aOff, off + j);
            if (v > best) { best = v; bestJ = j; }
        }
        st(outOff, i, best); store<i32>(idxOff + i * 4, bestJ + 1);
    }
}

export function minRows(aOff: i32, outOff: i32, idxOff: i32, R: i32, C: i32): void {
    for (let i: i32 = 0; i < R; i++) {
        const off: i32 = i * C;
        let best: f64 = ld(aOff, off); let bestJ: i32 = 0;
        for (let j: i32 = 1; j < C; j++) {
            const v: f64 = ld(aOff, off + j);
            if (v < best) { best = v; bestJ = j; }
        }
        st(outOff, i, best); store<i32>(idxOff + i * 4, bestJ + 1);
    }
}

// ─── 9. PROPERTY CHECKS ──────────────────────────────────────────────────────

export function isSymmetricF64(aOff: i32, n: i32, tol: f64): i32 {
    for (let i: i32 = 0; i < n; i++)
        for (let j: i32 = i + 1; j < n; j++)
            if (fAbs(ld(aOff, i * n + j) - ld(aOff, j * n + i)) > tol) return 0;
    return 1;
}

export function isUpperTriF64(aOff: i32, R: i32, C: i32, tol: f64): i32 {
    for (let i: i32 = 1; i < R; i++)
        for (let j: i32 = 0; j < iMin(i, C); j++)
            if (fAbs(ld(aOff, i * C + j)) > tol) return 0;
    return 1;
}

export function isLowerTriF64(aOff: i32, R: i32, C: i32, tol: f64): i32 {
    for (let i: i32 = 0; i < R; i++)
        for (let j: i32 = i + 1; j < C; j++)
            if (fAbs(ld(aOff, i * C + j)) > tol) return 0;
    return 1;
}

export function isDiagonalF64(aOff: i32, R: i32, C: i32, tol: f64): i32 {
    for (let i: i32 = 0; i < R; i++)
        for (let j: i32 = 0; j < C; j++)
            if (i != j && fAbs(ld(aOff, i * C + j)) > tol) return 0;
    return 1;
}

export function isZeroF64(aOff: i32, len: i32, tol: f64): i32 {
    for (let i: i32 = 0; i < len; i++) if (fAbs(ld(aOff, i)) > tol) return 0;
    return 1;
}

export function hasFiniteF64(aOff: i32, len: i32): i32 {
    for (let i: i32 = 0; i < len; i++) {
        const v: f64 = ld(aOff, i);
        if (v != v || v == Infinity || v == -Infinity) return 0;
    }
    return 1;
}

export function isDiagDomF64(aOff: i32, R: i32, C: i32): i32 {
    for (let i: i32 = 0; i < R; i++) {
        const off: i32 = i * C;
        let s: f64 = 0.0;
        for (let j: i32 = 0; j < C; j++) if (j != i) s += fAbs(ld(aOff, off + j));
        if (fAbs(ld(aOff, off + i)) < s) return 0;
    }
    return 1;
}

// ─── 10. SOLVER TRIANGOLARI ──────────────────────────────────────────────────

export function solveLower(lOff: i32, bOff: i32, xOff: i32, n: i32, bc: i32): void {
    for (let col: i32 = 0; col < bc; col++) {
        st(xOff + col * 8, 0, ld(bOff + col * 8, 0) / ld(lOff, 0));
        for (let i: i32 = 1; i < n; i++) {
            let s: f64 = load<f64>(bOff + (i * bc + col) * 8);
            for (let k: i32 = 0; k < i; k++)
                s -= ld(lOff, i * n + k) * load<f64>(xOff + (k * bc + col) * 8);
            store<f64>(xOff + (i * bc + col) * 8, s / ld(lOff, i * n + i));
        }
    }
}

export function solveLowerUnit(lOff: i32, bOff: i32, xOff: i32, n: i32, bc: i32): void {
    for (let col: i32 = 0; col < bc; col++) {
        store<f64>(xOff + col * 8, load<f64>(bOff + col * 8));
        for (let i: i32 = 1; i < n; i++) {
            let s: f64 = load<f64>(bOff + (i * bc + col) * 8);
            for (let k: i32 = 0; k < i; k++)
                s -= ld(lOff, i * n + k) * load<f64>(xOff + (k * bc + col) * 8);
            store<f64>(xOff + (i * bc + col) * 8, s);
        }
    }
}

export function solveUpper(uOff: i32, bOff: i32, xOff: i32, n: i32, bc: i32): void {
    const nm1: i32 = n - 1;
    for (let col: i32 = 0; col < bc; col++) {
        store<f64>(xOff + (nm1 * bc + col) * 8,
            load<f64>(bOff + (nm1 * bc + col) * 8) / ld(uOff, nm1 * n + nm1));
        for (let i: i32 = nm1 - 1; i >= 0; i--) {
            let s: f64 = load<f64>(bOff + (i * bc + col) * 8);
            for (let k: i32 = i + 1; k < n; k++)
                s -= ld(uOff, i * n + k) * load<f64>(xOff + (k * bc + col) * 8);
            store<f64>(xOff + (i * bc + col) * 8, s / ld(uOff, i * n + i));
        }
    }
}

// ─── 11. DECOMPOSIZIONE LUP ──────────────────────────────────────────────────

export function lupDecomp(wOff: i32, pOff: i32, n: i32): i32 {
    for (let i: i32 = 0; i < n; i++) store<i32>(pOff + i * 4, i);
    let swaps: i32 = 0;
    const EPS: f64 = 1e-12;
    for (let i: i32 = 0; i < n; i++) {
        let maxVal: f64 = 0.0; let maxRow: i32 = i;
        for (let k: i32 = i; k < n; k++) {
            const v: f64 = fAbs(ld(wOff, k * n + i));
            if (v > maxVal) { maxVal = v; maxRow = k; }
        }
        if (maxVal < EPS) return -1;
        if (maxRow != i) {
            swaps++;
            const iBase: i32 = wOff + i * n * 8;
            const mBase: i32 = wOff + maxRow * n * 8;
            for (let j: i32 = 0; j < n; j++) {
                const off: i32 = j * 8;
                const t: f64 = load<f64>(iBase + off);
                store<f64>(iBase + off, load<f64>(mBase + off));
                store<f64>(mBase + off, t);
            }
            const tp: i32 = load<i32>(pOff + i * 4);
            store<i32>(pOff + i * 4, load<i32>(pOff + maxRow * 4));
            store<i32>(pOff + maxRow * 4, tp);
        }
        const pivot: f64 = ld(wOff, i * n + i);
        for (let j: i32 = i + 1; j < n; j++) {
            const jOff: i32 = j * n;
            const factor: f64 = ld(wOff, jOff + i) / pivot;
            st(wOff, jOff + i, factor);
            const jRow: i32 = wOff + jOff * 8;
            const iRow: i32 = wOff + i * n * 8;
            let k: i32 = i + 1;
            const vf: v128 = splat2(factor);
            for (; k + 1 < n; k += 2) {
                const off: i32 = k * 8;
                st2(jRow, off, f64x2.sub(ld2(jRow, off), f64x2.mul(vf, ld2(iRow, off))));
            }
            if (k < n) {
                store<f64>(jRow + k * 8, load<f64>(jRow + k * 8) - factor * load<f64>(iRow + k * 8));
            }
        }
    }
    return swaps;
}

// ─── 12. CHOLESKY ────────────────────────────────────────────────────────────

export function choleskyDecomp(aOff: i32, lOff: i32, n: i32): i32 {
    zeroF64(lOff, n * n);
    for (let j: i32 = 0; j < n; j++) {
        let diagSum: f64 = 0.0;
        const jRow: i32 = lOff + j * n * 8;
        for (let k: i32 = 0; k < j; k++) { const v = load<f64>(jRow + k * 8); diagSum += v * v; }
        const d: f64 = ld(aOff, j * n + j) - diagSum;
        if (d < 0.0) return -1;
        const ljj: f64 = Math.sqrt(d);
        store<f64>(jRow + j * 8, ljj);
        for (let i: i32 = j + 1; i < n; i++) {
            const iRow: i32 = lOff + i * n * 8;
            let offSum: f64 = _dotN(iRow, jRow, j);
            store<f64>(iRow + j * 8, (ld(aOff, i * n + j) - offSum) / ljj);
        }
    }
    return 0;
}

// ─── 13. QR (MGS) ────────────────────────────────────────────────────────────

export function qrDecomp(wOff: i32, qOff: i32, rOff: i32, m: i32, n: i32): void {
    zeroF64(qOff, m * n);
    zeroF64(rOff, n * n);
    for (let k: i32 = 0; k < n; k++) {
        let normSq: f64 = 0.0;
        for (let i: i32 = 0; i < m; i++) { const v = ld(wOff, i * n + k); normSq += v * v; }
        const normK: f64 = Math.sqrt(normSq);
        st(rOff, k * n + k, normK);
        const invNorm: f64 = 1.0 / normK;
        for (let i: i32 = 0; i < m; i++) st(qOff, i * n + k, ld(wOff, i * n + k) * invNorm);
        for (let j: i32 = k + 1; j < n; j++) {
            let dot: f64 = 0.0;
            for (let i: i32 = 0; i < m; i++) dot += ld(qOff, i * n + k) * ld(wOff, i * n + j);
            st(rOff, k * n + j, dot);
            for (let i: i32 = 0; i < m; i++)
                st(wOff, i * n + j, ld(wOff, i * n + j) - dot * ld(qOff, i * n + k));
        }
    }
}

// ─── 14. LDLT ────────────────────────────────────────────────────────────────

export function ldltDecomp(aOff: i32, lOff: i32, dOff: i32, n: i32): i32 {
    zeroF64(lOff, n * n);
    for (let i: i32 = 0; i < n; i++) st(lOff, i * n + i, 1.0);
    zeroF64(dOff, n);
    for (let j: i32 = 0; j < n; j++) {
        let djj: f64 = ld(aOff, j * n + j);
        for (let k: i32 = 0; k < j; k++) {
            const ljk: f64 = ld(lOff, j * n + k);
            djj -= ljk * ljk * ld(dOff, k);
        }
        if (fAbs(djj) < 1e-14) return -1;
        st(dOff, j, djj);
        for (let i: i32 = j + 1; i < n; i++) {
            let lij: f64 = ld(aOff, i * n + j);
            for (let k: i32 = 0; k < j; k++)
                lij -= ld(lOff, i * n + k) * ld(lOff, j * n + k) * ld(dOff, k);
            st(lOff, i * n + j, lij / djj);
        }
    }
    return 0;
}

// ─── 15. SOLVER ITERATIVI ─────────────────────────────────────────────────────
// Ottimizzazioni applicate a tutti:
//   • Loop j<i e j>i separati: zero branch nel loop SIMD
//   • Convergenza relativa: ||dx||_inf / max(||x||_inf, 1) < tol
//
// Jacobi e JOR aggiungono:
//   • diagInv precompilato: n mul invece di n div per iterazione esterna

// Jacobi: diagInvOff = workspace n elementi per diagInv.
// Ritorna iterazioni usate (>0) o -1 (non converge in maxIter).
export function jacobiSolve(
    aOff: i32, bOff: i32, xOff: i32, xNewOff: i32, diagInvOff: i32,
    n: i32, tol: f64, maxIter: i32
): i32 {
    // Pre-pass: calcola diagInv[i] = 1/A[i,i]
    // Trasforma n divisioni/iterazione in n moltiplicazioni (~5x piu veloce su FP)
    for (let i: i32 = 0; i < n; i++) {
        st(diagInvOff, i, 1.0 / ld(aOff, i * n + i));
    }
    zeroF64(xOff, n);

    for (let iter: i32 = 0; iter < maxIter; iter++) {
        let maxDiff: f64 = 0.0;
        let maxAbsX: f64 = 0.0;

        for (let i: i32 = 0; i < n; i++) {
            const rowOff: i32 = aOff + i * n * 8;
            const di: f64 = ld(diagInvOff, i);

            // sum_{j<i} A[i,j]*x_old[j]  (SIMD, no branch)
            let s: f64 = 0.0;
            let j: i32 = 0;
            for (; j + 1 < i; j += 2) {
                const off: i32 = j * 8;
                const av: v128 = f64x2.mul(ld2(rowOff, off), ld2(xOff, off));
                s += f64x2.extract_lane(av, 0) + f64x2.extract_lane(av, 1);
            }
            for (; j < i; j++) s += ld(aOff, i * n + j) * ld(xOff, j);

            // sum_{j>i} A[i,j]*x_old[j]  (SIMD, no branch)
            j = i + 1;
            for (; j + 1 < n; j += 2) {
                const off: i32 = j * 8;
                const av: v128 = f64x2.mul(ld2(rowOff, off), ld2(xOff, off));
                s += f64x2.extract_lane(av, 0) + f64x2.extract_lane(av, 1);
            }
            for (; j < n; j++) s += ld(aOff, i * n + j) * ld(xOff, j);

            // x_new[i] = (b[i] - s) / A[i,i]  = (b[i] - s) * diagInv[i]
            const xi: f64 = (ld(bOff, i) - s) * di;
            st(xNewOff, i, xi);

            const diff: f64 = fAbs(xi - ld(xOff, i));
            if (diff > maxDiff) maxDiff = diff;
            const ax: f64 = fAbs(xi);
            if (ax > maxAbsX) maxAbsX = ax;
        }

        // Swap: copia xNew -> x (ping-pong: garantisce x_old per tutti j)
        for (let i: i32 = 0; i < n; i++) st(xOff, i, ld(xNewOff, i));

        // Convergenza relativa: invariante rispetto alla scala
        const denom: f64 = maxAbsX > 1.0 ? maxAbsX : 1.0;
        if (maxDiff / denom < tol) return iter + 1;
    }
    return -1;
}

// Gauss-Seidel: loop separati j<i (x aggiornato) e j>i (x old), no branch SIMD.
export function gaussSeidelSolve(
    aOff: i32, bOff: i32, xOff: i32,
    n: i32, tol: f64, maxIter: i32
): i32 {
    zeroF64(xOff, n);
    for (let iter: i32 = 0; iter < maxIter; iter++) {
        let maxDiff: f64 = 0.0;
        let maxAbsX: f64 = 0.0;

        for (let i: i32 = 0; i < n; i++) {
            const rowOff: i32 = aOff + i * n * 8;

            // j < i: usa x AGGIORNATO (GS: x^{k+1} per j < i)  — SIMD senza branch
            let s: f64 = 0.0;
            let j: i32 = 0;
            for (; j + 1 < i; j += 2) {
                const off: i32 = j * 8;
                const v: v128 = f64x2.mul(ld2(rowOff, off), ld2(xOff, off));
                s += f64x2.extract_lane(v, 0) + f64x2.extract_lane(v, 1);
            }
            for (; j < i; j++) s += ld(aOff, i * n + j) * ld(xOff, j);

            // j > i: usa x OLD (non ancora aggiornato)  — SIMD senza branch
            j = i + 1;
            for (; j + 1 < n; j += 2) {
                const off: i32 = j * 8;
                const v: v128 = f64x2.mul(ld2(rowOff, off), ld2(xOff, off));
                s += f64x2.extract_lane(v, 0) + f64x2.extract_lane(v, 1);
            }
            for (; j < n; j++) s += ld(aOff, i * n + j) * ld(xOff, j);

            const xi: f64 = (ld(bOff, i) - s) / ld(aOff, i * n + i);
            const diff: f64 = fAbs(xi - ld(xOff, i));
            st(xOff, i, xi);
            if (diff > maxDiff) maxDiff = diff;
            const ax: f64 = fAbs(xi);
            if (ax > maxAbsX) maxAbsX = ax;
        }

        const denom: f64 = maxAbsX > 1.0 ? maxAbsX : 1.0;
        if (maxDiff / denom < tol) return iter + 1;
    }
    return -1;
}

// SOR: j<i (x aggiornato) e j>i (x old), no branch SIMD.
export function sorSolve(
    aOff: i32, bOff: i32, xOff: i32,
    n: i32, omega: f64, tol: f64, maxIter: i32
): i32 {
    zeroF64(xOff, n);
    const oneMinOmega: f64 = 1.0 - omega;
    for (let iter: i32 = 0; iter < maxIter; iter++) {
        let maxDiff: f64 = 0.0;
        let maxAbsX: f64 = 0.0;
        for (let i: i32 = 0; i < n; i++) {
            const rowOff: i32 = aOff + i * n * 8;
            let s: f64 = 0.0;
            let j: i32 = 0;
            // j < i: x aggiornato (SOR GS-style)  — SIMD senza branch
            for (; j + 1 < i; j += 2) {
                const off: i32 = j * 8;
                const v: v128 = f64x2.mul(ld2(rowOff, off), ld2(xOff, off));
                s += f64x2.extract_lane(v, 0) + f64x2.extract_lane(v, 1);
            }
            for (; j < i; j++) s += ld(aOff, i * n + j) * ld(xOff, j);
            // j > i: x old  — SIMD senza branch
            j = i + 1;
            for (; j + 1 < n; j += 2) {
                const off: i32 = j * 8;
                const v: v128 = f64x2.mul(ld2(rowOff, off), ld2(xOff, off));
                s += f64x2.extract_lane(v, 0) + f64x2.extract_lane(v, 1);
            }
            for (; j < n; j++) s += ld(aOff, i * n + j) * ld(xOff, j);
            const gsStep: f64 = (ld(bOff, i) - s) / ld(aOff, i * n + i);
            const xi: f64 = oneMinOmega * ld(xOff, i) + omega * gsStep;
            const diff: f64 = fAbs(xi - ld(xOff, i));
            st(xOff, i, xi);
            if (diff > maxDiff) maxDiff = diff;
            const ax: f64 = fAbs(xi);
            if (ax > maxAbsX) maxAbsX = ax;
        }
        const denom: f64 = maxAbsX > 1.0 ? maxAbsX : 1.0;
        if (maxDiff / denom < tol) return iter + 1;
    }
    return -1;
}

// CG (Gradiente Coniugato): A SPD.
export function cgSolve(
    aOff: i32, bOff: i32, xOff: i32,
    rOff: i32, pOff: i32, apOff: i32,
    n: i32, tol: f64, maxIter: i32
): i32 {
    zeroF64(xOff, n);
    for (let i: i32 = 0; i < n; i++) {
        const bv: f64 = ld(bOff, i);
        st(rOff, i, bv);
        st(pOff, i, bv);
    }
    let rho: f64 = _dotN(rOff, rOff, n);
    const tol2: f64 = tol * tol;
    for (let iter: i32 = 0; iter < maxIter; iter++) {
        if (rho < tol2) return iter;
        for (let i: i32 = 0; i < n; i++)
            st(apOff, i, _dotN(aOff + i * n * 8, pOff, n));
        const pAp: f64 = _dotN(pOff, apOff, n);
        if (fAbs(pAp) < 1e-300) return iter;
        const alpha: f64 = rho / pAp;
        const valpha: v128 = splat2(alpha);
        let i: i32 = 0;
        for (; i + 1 < n; i += 2) {
            const off: i32 = i << 3;
            st2(xOff, off, f64x2.add(ld2(xOff, off), f64x2.mul(valpha, ld2(pOff, off))));
            st2(rOff, off, f64x2.sub(ld2(rOff, off), f64x2.mul(valpha, ld2(apOff, off))));
        }
        if (i < n) {
            st(xOff, i, ld(xOff, i) + alpha * ld(pOff, i));
            st(rOff, i, ld(rOff, i) - alpha * ld(apOff, i));
        }
        const rhoNew: f64 = _dotN(rOff, rOff, n);
        const beta: f64 = rhoNew / rho;
        const vbeta: v128 = splat2(beta);
        rho = rhoNew;
        i = 0;
        for (; i + 1 < n; i += 2) {
            const off: i32 = i << 3;
            st2(pOff, off, f64x2.add(ld2(rOff, off), f64x2.mul(vbeta, ld2(pOff, off))));
        }
        if (i < n) st(pOff, i, ld(rOff, i) + beta * ld(pOff, i));
    }
    return -1;
}

// ─── 16. JOR (Jacobi Over-Relaxation) ────────────────────────────────────────
// x_i^{k+1} = (1-omega)*x_i^k + omega*(b_i - sum_{j!=i} A_ij*x_j^k) / A_ii
// Usa x^k (old) per TUTTI j (a differenza di SOR che usa x^{k+1} per j<i).
// diagInvOff: workspace n elementi per diagInv precompilato.
export function jorSolve(
    aOff: i32, bOff: i32, xOff: i32, xNewOff: i32, diagInvOff: i32,
    n: i32, omega: f64, tol: f64, maxIter: i32
): i32 {
    for (let i: i32 = 0; i < n; i++) {
        st(diagInvOff, i, 1.0 / ld(aOff, i * n + i));
    }
    zeroF64(xOff, n);
    const oneMinOmega: f64 = 1.0 - omega;

    for (let iter: i32 = 0; iter < maxIter; iter++) {
        let maxDiff: f64 = 0.0;
        let maxAbsX: f64 = 0.0;

        for (let i: i32 = 0; i < n; i++) {
            const rowOff: i32 = aOff + i * n * 8;
            const di: f64 = ld(diagInvOff, i);
            let s: f64 = 0.0;
            let j: i32 = 0;
            for (; j + 1 < i; j += 2) {
                const off: i32 = j * 8;
                const v: v128 = f64x2.mul(ld2(rowOff, off), ld2(xOff, off));
                s += f64x2.extract_lane(v, 0) + f64x2.extract_lane(v, 1);
            }
            for (; j < i; j++) s += ld(aOff, i * n + j) * ld(xOff, j);
            j = i + 1;
            for (; j + 1 < n; j += 2) {
                const off: i32 = j * 8;
                const v: v128 = f64x2.mul(ld2(rowOff, off), ld2(xOff, off));
                s += f64x2.extract_lane(v, 0) + f64x2.extract_lane(v, 1);
            }
            for (; j < n; j++) s += ld(aOff, i * n + j) * ld(xOff, j);

            const jacobiStep: f64 = (ld(bOff, i) - s) * di;
            const xi: f64 = oneMinOmega * ld(xOff, i) + omega * jacobiStep;
            st(xNewOff, i, xi);

            const diff: f64 = fAbs(xi - ld(xOff, i));
            if (diff > maxDiff) maxDiff = diff;
            const ax: f64 = fAbs(xi);
            if (ax > maxAbsX) maxAbsX = ax;
        }

        for (let i: i32 = 0; i < n; i++) st(xOff, i, ld(xNewOff, i));

        const denom: f64 = maxAbsX > 1.0 ? maxAbsX : 1.0;
        if (maxDiff / denom < tol) return iter + 1;
    }
    return -1;
}

// ============================================================
// src/gpu/shaders.ts
//
// Compute shader WGSL per operazioni matriciali.
//
// IMPORTANTE — Precisione:
//   WebGPU supporta nativamente solo f32 (32-bit float).
//   f64 (double) non è disponibile come tipo nativo in WGSL
//   (la specifica WebGPU non lo prevede ancora, a differenza di Metal/Vulkan).
//   → Tutte le operazioni GPU usano f32.
//   → Per n > GPU_THRESHOLD, la precisione è ~7 cifre significative (single).
//   → Usare solo dove la velocità supera il requisito di precisione.
//   → setGPUEnabled(false) disabilita il backend GPU globalmente.
//
// Workgroup sizes:
//   matmul       : 16×16 = 256 threads (tile di shared memory)
//   elementwise  : 256 threads lineari
//   jacobi iter  : 256 threads, una riga per thread
// ============================================================

// ─── Matmul a tiles 16×16 ────────────────────────────────────────────────────
// C = A × B   (f32, A: M×K, B: K×N, C: M×N, tutti row-major)
// Ogni workgroup calcola un tile 16×16 di C.
// Sfrutta shared memory (var<workgroup>) per ridurre accessi globali.
export const SHADER_MATMUL = /* wgsl */`
struct Dims {
    M : u32,
    K : u32,
    N : u32,
    _pad : u32,
}
@group(0) @binding(0) var<uniform>             dims : Dims;
@group(0) @binding(1) var<storage, read>       A    : array<f32>;
@group(0) @binding(2) var<storage, read>       B    : array<f32>;
@group(0) @binding(3) var<storage, read_write> C    : array<f32>;

const TILE : u32 = 16u;

var<workgroup> tileA : array<array<f32, 16>, 16>;
var<workgroup> tileB : array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid  : vec3<u32>,
    @builtin(local_invocation_id)  lid  : vec3<u32>,
    @builtin(workgroup_id)         wgid : vec3<u32>
) {
    let row = gid.y;
    let col = gid.x;
    let lRow = lid.y;
    let lCol = lid.x;

    var acc : f32 = 0.0;
    let numTiles = (dims.K + TILE - 1u) / TILE;

    for (var t : u32 = 0u; t < numTiles; t++) {
        let aCol = t * TILE + lCol;
        let bRow = t * TILE + lRow;

        // Carica tile A in shared memory
        if (row < dims.M && aCol < dims.K) {
            tileA[lRow][lCol] = A[row * dims.K + aCol];
        } else {
            tileA[lRow][lCol] = 0.0;
        }

        // Carica tile B in shared memory
        if (bRow < dims.K && col < dims.N) {
            tileB[lRow][lCol] = B[bRow * dims.N + col];
        } else {
            tileB[lRow][lCol] = 0.0;
        }

        workgroupBarrier();

        // Prodotto parziale del tile
        for (var k : u32 = 0u; k < TILE; k++) {
            acc += tileA[lRow][k] * tileB[k][lCol];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = acc;
    }
}
`;

// ─── Element-wise operations ──────────────────────────────────────────────────
// Operazione: C[i] = A[i] op B[i]   (f32, len elementi)
// op è selezionato da un uniforme:  0=add, 1=sub, 2=mul, 3=div
export const SHADER_ELEMENTWISE = /* wgsl */`
struct Params {
    len : u32,
    op  : u32,   // 0=add, 1=sub, 2=mul, 3=div
    _p0 : u32,
    _p1 : u32,
}
@group(0) @binding(0) var<uniform>             p : Params;
@group(0) @binding(1) var<storage, read>       A : array<f32>;
@group(0) @binding(2) var<storage, read>       B : array<f32>;
@group(0) @binding(3) var<storage, read_write> C : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    let a = A[i]; let b = B[i];
    switch (p.op) {
        case 0u: { C[i] = a + b; }
        case 1u: { C[i] = a - b; }
        case 2u: { C[i] = a * b; }
        case 3u: { C[i] = a / b; }
        default: { C[i] = a; }
    }
}
`;

// ─── Scalar operation ─────────────────────────────────────────────────────────
// C[i] = A[i] op scalar
export const SHADER_SCALAR_OP = /* wgsl */`
struct Params {
    len    : u32,
    op     : u32,
    _pad0  : u32,
    _pad1  : u32,
    scalar : f32,
    _pad2  : f32,
    _pad3  : f32,
    _pad4  : f32,
}
@group(0) @binding(0) var<uniform>             p : Params;
@group(0) @binding(1) var<storage, read>       A : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    let a = A[i]; let s = p.scalar;
    switch (p.op) {
        case 0u: { C[i] = a + s; }
        case 1u: { C[i] = a - s; }
        case 2u: { C[i] = a * s; }
        case 3u: { C[i] = a / s; }
        case 4u: { C[i] = pow(a, s); }
        default: { C[i] = a; }
    }
}
`;

// ─── Unary operations ─────────────────────────────────────────────────────────
// C[i] = f(A[i])
// op: 0=abs, 1=neg, 2=sqrt, 3=exp, 4=sin, 5=cos, 6=tan, 7=floor, 8=ceil, 9=round
export const SHADER_UNARY = /* wgsl */`
struct Params {
    len : u32,
    op  : u32,
    _p0 : u32,
    _p1 : u32,
}
@group(0) @binding(0) var<uniform>             p : Params;
@group(0) @binding(1) var<storage, read>       A : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= p.len) { return; }
    let a = A[i];
    switch (p.op) {
        case 0u: { C[i] = abs(a); }
        case 1u: { C[i] = -a; }
        case 2u: { C[i] = sqrt(max(a, 0.0)); }
        case 3u: { C[i] = exp(a); }
        case 4u: { C[i] = sin(a); }
        case 5u: { C[i] = cos(a); }
        case 6u: { C[i] = tan(a); }
        case 7u: { C[i] = floor(a); }
        case 8u: { C[i] = ceil(a); }
        case 9u: { C[i] = round(a); }
        default: { C[i] = a; }
    }
}
`;

// ─── Riduzione: somma totale (2 passate) ─────────────────────────────────────
export const SHADER_REDUCE_SUM = /* wgsl */`
struct Params {
    len     : u32,
    outLen  : u32,
    _p0     : u32,
    _p1     : u32,
}
@group(0) @binding(0) var<uniform>             p   : Params;
@group(0) @binding(1) var<storage, read>       src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;

var<workgroup> shared : array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid  : vec3<u32>,
    @builtin(local_invocation_id)  lid  : vec3<u32>,
    @builtin(workgroup_id)         wgid : vec3<u32>
) {
    let i = gid.x;
    let li = lid.x;
    shared[li] = select(0.0, src[i], i < p.len);
    workgroupBarrier();

    // Tree reduction
    var stride : u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (li < stride && li + stride < 256u) {
            shared[li] += shared[li + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (li == 0u) {
        dst[wgid.x] = shared[0];
    }
}
`;

// ─── Jacobi iteration (un'iterazione completa) ───────────────────────────────
// Ogni thread calcola x_new[i] per una riga.
// x_old è in lettura, x_new è scrittura, A e b sono read-only.
// diagInv[i] = 1/A[i,i] precompilato.
// outDiff[i] = |x_new[i] - x_old[i]|  per la convergenza.
export const SHADER_JACOBI = /* wgsl */`
struct Params {
    n   : u32,
    _p0 : u32,
    _p1 : u32,
    _p2 : u32,
}
@group(0) @binding(0) var<uniform>             p       : Params;
@group(0) @binding(1) var<storage, read>       A       : array<f32>;
@group(0) @binding(2) var<storage, read>       b       : array<f32>;
@group(0) @binding(3) var<storage, read>       xOld    : array<f32>;
@group(0) @binding(4) var<storage, read>       diagInv : array<f32>;
@group(0) @binding(5) var<storage, read_write> xNew    : array<f32>;
@group(0) @binding(6) var<storage, read_write> outDiff : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }

    let rowBase = i * p.n;
    var s : f32 = 0.0;

    // sum_{j != i} A[i,j] * x_old[j]
    for (var j : u32 = 0u; j < i; j++) {
        s += A[rowBase + j] * xOld[j];
    }
    for (var j : u32 = i + 1u; j < p.n; j++) {
        s += A[rowBase + j] * xOld[j];
    }

    let xi = (b[i] - s) * diagInv[i];
    xNew[i] = xi;
    outDiff[i] = abs(xi - xOld[i]);
}
`;

// ─── Norm Frobenius: riduzione somma quadrati ─────────────────────────────────
export const SHADER_NORM_FRO = /* wgsl */`
struct Params {
    len : u32,
    _p0 : u32, _p1 : u32, _p2 : u32,
}
@group(0) @binding(0) var<uniform>             p   : Params;
@group(0) @binding(1) var<storage, read>       src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;

var<workgroup> shared : array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid  : vec3<u32>,
    @builtin(local_invocation_id)  lid  : vec3<u32>,
    @builtin(workgroup_id)         wgid : vec3<u32>
) {
    let i  = gid.x;
    let li = lid.x;
    let v  = select(0.0, src[i], i < p.len);
    shared[li] = v * v;
    workgroupBarrier();
    var stride : u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (li < stride) { shared[li] += shared[li + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (li == 0u) { dst[wgid.x] = shared[0]; }
}
`;

// ============================================================
// src/wasm/wasm_bridge.ts — Bridge TypeScript ↔ WebAssembly
// ============================================================

// ── SOGLIE MINIME DI ATTIVAZIONE WASM ─────────────────────────────────────────
// Per matrici piccole, l'overhead di serializzazione JS→WASM→JS supera
// il guadagno computazionale. Queste soglie calibrate empiricamente
// garantiscono che WASM venga usato solo dove porta beneficio netto.
//
// Modello overhead:
//   costo_totale_wasm = T_copy_in + T_kernel + T_copy_out
//   T_copy_in  = n_elems × ~10ns  (lettura .value + store f64)
//   T_copy_out = n_elems × ~8ns   (load f64 + fromNumber())
//   T_kernel   = f(n_elems, complexity)
//
//   Breakeven element-wise O(n²): T_copy > T_js → n_elems ≈ 512  (n ≈ 23)
//   Breakeven matmul O(n³):       breakeven molto prima → n_elems ≈ 256 (n ≈ 16)
//   Breakeven decomp O(n³):       n_elems ≈ 256 (n ≈ 16)
//   Breakeven iterativi:          costo fisso WASM < 1 iter JS → n ≈ 8
//
export const WASM_THRESHOLD = {
    /** Element-wise ops (add, sub, dotMul, unary): n² ≥ threshold */
    ELEMENTWISE : 512,   // n ≥ 23
    /** Matmul O(n³): n² ≥ threshold (ammortizzato meglio) */
    MATMUL      : 256,   // n ≥ 16
    /** Decomposizioni O(n³) */
    DECOMP      : 256,   // n ≥ 16
    /** Property checks O(n²) */
    PROPERTY    : 256,   // n ≥ 16
    /** Solver triangolari O(n²) */
    TRIANGULAR  : 256,   // n ≥ 16
    /** Solver iterativi: loop intero in WASM, overhead fisso → soglia bassa */
    ITERATIVE   : 64,    // n ≥ 8
    /** Norme e statistiche O(n²) */
    STATS       : 256,   // n ≥ 16
} as const;

interface WasmExports {
    memory: WebAssembly.Memory;
    // zero-fill
    zeroF64      : (ptr: number, len: number) => void;
    // element-wise
    addMatrix    : (aOff: number, bOff: number, cOff: number, len: number) => void;
    subMatrix    : (aOff: number, bOff: number, cOff: number, len: number) => void;
    dotMul       : (aOff: number, bOff: number, cOff: number, len: number) => void;
    dotDiv       : (aOff: number, bOff: number, cOff: number, len: number) => void;
    addScalar    : (aOff: number, cOff: number, len: number, scalar: number) => void;
    subScalar    : (aOff: number, cOff: number, len: number, scalar: number) => void;
    mulScalar    : (aOff: number, cOff: number, len: number, scalar: number) => void;
    dotPowScalar : (aOff: number, cOff: number, len: number, exp: number) => void;
    // broadcast
    addRowVec    : (aOff: number, bOff: number, cOff: number, R: number, C: number) => void;
    subRowVec    : (aOff: number, bOff: number, cOff: number, R: number, C: number) => void;
    addColVec    : (aOff: number, bOff: number, cOff: number, R: number, C: number) => void;
    subColVec    : (aOff: number, bOff: number, cOff: number, R: number, C: number) => void;
    dotMulRowVec : (aOff: number, bOff: number, cOff: number, R: number, C: number) => void;
    dotMulColVec : (aOff: number, bOff: number, cOff: number, R: number, C: number) => void;
    // matmul + matvec
    matmul       : (aOff: number, bOff: number, cOff: number, M: number, K: number, N: number) => void;
    matvec       : (aOff: number, xOff: number, yOff: number, n: number) => void;
    // transpose
    transpose    : (aOff: number, cOff: number, R: number, C: number) => void;
    // norme
    normFro      : (aOff: number, len: number) => number;
    normVec1     : (aOff: number, len: number) => number;
    normVecInf   : (aOff: number, len: number) => number;
    normMat1     : (aOff: number, R: number, C: number) => number;
    normMatInf   : (aOff: number, R: number, C: number) => number;
    // unary
    unaryAbs     : (aOff: number, cOff: number, len: number) => void;
    unaryNeg     : (aOff: number, cOff: number, len: number) => void;
    unarySqrt    : (aOff: number, cOff: number, len: number) => void;
    unaryRound   : (aOff: number, cOff: number, len: number) => void;
    unaryFloor   : (aOff: number, cOff: number, len: number) => void;
    unaryCeil    : (aOff: number, cOff: number, len: number) => void;
    unaryExp     : (aOff: number, cOff: number, len: number) => void;
    unarySin     : (aOff: number, cOff: number, len: number) => void;
    unaryCos     : (aOff: number, cOff: number, len: number) => void;
    unaryTan     : (aOff: number, cOff: number, len: number) => void;
    // statistics
    totalSum     : (aOff: number, len: number) => number;
    trace        : (aOff: number, n: number) => number;
    sumCols      : (aOff: number, outOff: number, R: number, C: number) => void;
    sumRows      : (aOff: number, outOff: number, R: number, C: number) => void;
    maxCols      : (aOff: number, outOff: number, idxOff: number, R: number, C: number) => void;
    minCols      : (aOff: number, outOff: number, idxOff: number, R: number, C: number) => void;
    maxRows      : (aOff: number, outOff: number, idxOff: number, R: number, C: number) => void;
    minRows      : (aOff: number, outOff: number, idxOff: number, R: number, C: number) => void;
    // property checks (return i32: 1=true, 0=false)
    isSymmetricF64 : (aOff: number, n: number, tol: number) => number;
    isUpperTriF64  : (aOff: number, R: number, C: number, tol: number) => number;
    isLowerTriF64  : (aOff: number, R: number, C: number, tol: number) => number;
    isDiagonalF64  : (aOff: number, R: number, C: number, tol: number) => number;
    isZeroF64      : (aOff: number, len: number, tol: number) => number;
    hasFiniteF64   : (aOff: number, len: number) => number;
    isDiagDomF64   : (aOff: number, R: number, C: number) => number;
    // triangular solvers
    solveLower     : (lOff: number, bOff: number, xOff: number, n: number, bc: number) => void;
    solveLowerUnit : (lOff: number, bOff: number, xOff: number, n: number, bc: number) => void;
    solveUpper     : (uOff: number, bOff: number, xOff: number, n: number, bc: number) => void;
    // decompositions
    lupDecomp      : (wOff: number, pOff: number, n: number) => number;
    choleskyDecomp : (aOff: number, lOff: number, n: number) => number;
    qrDecomp       : (wOff: number, qOff: number, rOff: number, m: number, n: number) => void;
    ldltDecomp     : (aOff: number, lOff: number, dOff: number, n: number) => number;
    // iterative solvers
    jacobiSolve      : (aOff: number, bOff: number, xOff: number, xNewOff: number, diagInvOff: number, n: number, tol: number, maxIter: number) => number;
    gaussSeidelSolve : (aOff: number, bOff: number, xOff: number, n: number, tol: number, maxIter: number) => number;
    sorSolve         : (aOff: number, bOff: number, xOff: number, n: number, omega: number, tol: number, maxIter: number) => number;
    cgSolve          : (aOff: number, bOff: number, xOff: number, rOff: number, pOff: number, apOff: number, n: number, tol: number, maxIter: number) => number;
    jorSolve         : (aOff: number, bOff: number, xOff: number, xNewOff: number, diagInvOff: number, n: number, omega: number, tol: number, maxIter: number) => number;
}

const INITIAL_PAGES = 256;   // 16 MB
const MAX_PAGES     = 4096;  // 256 MB

function _abort(msg: number, file: number, line: number, col: number): never {
    throw new Error(`[WASM abort] line=${line} col=${col}`);
}
function _seed(): number { return Math.random() * 0xFFFF_FFFF; }

export class WasmBridge {
    private static _instance: WasmBridge | null = null;
    private static _initPromise: Promise<WasmBridge> | null = null;

    readonly exports: WasmExports;
    private heapPtr  = 0;
    private heap     : Float64Array;
    private heapRaw  : Uint8Array;
    private heapCap  : number;

    private constructor(instance: WebAssembly.Instance) {
        this.exports = instance.exports as unknown as WasmExports;
        const mem    = this.exports.memory;
        this.heap    = new Float64Array(mem.buffer);
        this.heapRaw = new Uint8Array(mem.buffer);
        this.heapCap = mem.buffer.byteLength;
    }

    static async getInstance(): Promise<WasmBridge> {
        if (WasmBridge._instance) return WasmBridge._instance;
        if (!WasmBridge._initPromise) WasmBridge._initPromise = WasmBridge._load();
        return WasmBridge._initPromise;
    }

    private static async _load(): Promise<WasmBridge> {
        const memory = new WebAssembly.Memory({ initial: INITIAL_PAGES, maximum: MAX_PAGES });
        let wasmBytes: BufferSource;

        if (typeof process !== "undefined" && process.versions?.node) {
            const { readFileSync }     = await import("fs");
            const { fileURLToPath }    = await import("url");
            const { dirname, resolve } = await import("path");
            const __filename = fileURLToPath(import.meta.url);
            const wasmPath   = resolve(dirname(__filename), "matrix_ops.wasm");
            wasmBytes = readFileSync(wasmPath);
        } else {
            wasmBytes = await (await fetch(new URL("./matrix_ops.wasm", import.meta.url).href)).arrayBuffer();
        }

        const { instance } = await WebAssembly.instantiate(wasmBytes, {
            env: { memory, abort: _abort, seed: _seed },
        });

        const bridge = new WasmBridge(instance);
        WasmBridge._instance = bridge;
        return bridge;
    }

    // ── allocazione bump ────────────────────────────────────────────────────
    alloc(nElems: number): number {
        const byteSize = nElems * 8;
        const aligned  = (this.heapPtr + 7) & ~7;
        if (aligned + byteSize > this.heapCap) this._grow(aligned + byteSize);
        this.heapPtr = aligned + byteSize;
        return aligned;
    }

    allocI32(nElems: number): number {
        const byteSize = nElems * 4;
        const aligned  = (this.heapPtr + 3) & ~3;
        if (aligned + byteSize > this.heapCap) this._grow(aligned + byteSize);
        this.heapPtr = aligned + byteSize;
        return aligned;
    }

    allocOutput(nElems: number): number { return this.alloc(nElems); }
    reset(): void { this.heapPtr = 0; }

    // ── I/O ─────────────────────────────────────────────────────────────────
    writeFloat64M(data: Array<{ value: number }>): number {
        const ptr = this.alloc(data.length);
        this._refresh();
        const idx = ptr >> 3;
        for (let i = 0; i < data.length; i++) this.heap[idx + i] = data[i].value;
        return ptr;
    }

    readF64(ptr: number, nElems: number): Float64Array {
        this._refresh();
        return this.heap.slice(ptr >> 3, (ptr >> 3) + nElems);
    }

    readPermutation(ptr: number, n: number): number[] {
        return Array.from(new Int32Array(this.exports.memory.buffer, ptr, n));
    }

    readI32Array(ptr: number, n: number): Int32Array {
        return new Int32Array(this.exports.memory.buffer, ptr, n).slice();
    }

    // ── private ─────────────────────────────────────────────────────────────
    private _refresh(): void {
        if (this.heap.buffer !== this.exports.memory.buffer) {
            this.heap    = new Float64Array(this.exports.memory.buffer);
            this.heapRaw = new Uint8Array(this.exports.memory.buffer);
            this.heapCap = this.exports.memory.buffer.byteLength;
        }
    }

    private _grow(needed: number): void {
        const extra = Math.ceil((needed - this.exports.memory.buffer.byteLength) / 65536) + 1;
        this.exports.memory.grow(extra);
        this._refresh();
    }
}

let _syncBridge: WasmBridge | null = null;
export function getBridgeSync(): WasmBridge | null {
    return _syncBridge ?? (WasmBridge as any)["_instance"] ?? null;
}

export async function initWasm(): Promise<void> {
    _syncBridge = await WasmBridge.getInstance();
}

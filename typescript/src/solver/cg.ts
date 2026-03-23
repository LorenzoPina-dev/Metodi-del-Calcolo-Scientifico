// solver/cg.ts — soglia WASM calibrata
import { Matrix }        from "..";
import { INumeric }      from "../type";
import { getBridgeSync, WASM_THRESHOLD } from "../wasm/wasm_bridge";

export function solveCG<T extends INumeric<T>>(
    A: Matrix<T>, b: Matrix<T>,
    tol = 1e-10, maxIter = 2000
): Matrix<T> {
    const n = A.rows;
    if (n !== A.cols) throw new Error("solveCG: matrice non quadrata.");

    if (A.isFloat64 && n * n >= WASM_THRESHOLD.ITERATIVE) {
        const w = getBridgeSync();
        if (w) {
            const aPtr  = w.writeFloat64M(A.data as any);
            const bPtr  = w.writeFloat64M(b.data as any);
            const xPtr  = w.allocOutput(n);
            const rPtr  = w.allocOutput(n);
            const pPtr  = w.allocOutput(n);
            const apPtr = w.allocOutput(n);
            w.exports.cgSolve(aPtr, bPtr, xPtr, rPtr, pPtr, apPtr, n, tol, maxIter);
            const flat = w.readF64(xPtr, n);
            w.reset();
            const out = A.like(n, 1);
            for (let i=0;i<n;i++) out.data[i] = A.zero.fromNumber(flat[i]);
            return out;
        }
        return _cgF64(A as any, b as any, tol, maxIter);
    }
    return _cgGeneric(A, b, tol, maxIter);
}

function _cgF64(A: Matrix<any>, b: Matrix<any>, tol: number, maxIter: number): Matrix<any> {
    const n = A.rows, ad = A.data;
    const x=new Float64Array(n), r=new Float64Array(n), p=new Float64Array(n), Ap=new Float64Array(n);
    for (let i=0;i<n;i++) r[i]=p[i]=(b.data[i] as any).value;
    let rho=_dot(r,r,n); const tol2=tol*tol;
    for (let iter=0;iter<maxIter;iter++) {
        if (rho<tol2) break;
        _matvec(ad,p,Ap,n);
        const pAp=_dot(p,Ap,n); if(Math.abs(pAp)<1e-300) break;
        const alpha=rho/pAp;
        for (let i=0;i<n;i++){x[i]+=alpha*p[i];r[i]-=alpha*Ap[i];}
        const rhoNew=_dot(r,r,n); const beta=rhoNew/rho; rho=rhoNew;
        for (let i=0;i<n;i++) p[i]=r[i]+beta*p[i];
    }
    const out = A.like(n,1);
    for (let i=0;i<n;i++) out.data[i]=A.zero.fromNumber(x[i]);
    return out;
}

function _dot(u: Float64Array, v: Float64Array, n: number): number {
    let s=0; for(let i=0;i<n;i++) s+=u[i]*v[i]; return s;
}
function _matvec(ad: any[], p: Float64Array, out: Float64Array, n: number): void {
    out.fill(0);
    for (let i=0;i<n;i++) { const off=i*n; let s=0; for(let j=0;j<n;j++) s+=(ad[off+j] as any).value*p[j]; out[i]=s; }
}

function _cgGeneric<T extends INumeric<T>>(A: Matrix<T>, b: Matrix<T>, tol: number, maxIter: number): Matrix<T> {
    const n=A.rows; let x=A.like(n,1), r=b.clone() as Matrix<T>, p=b.clone() as Matrix<T>;
    let rho=_dotGen(r,r); const tol2=tol*tol;
    for (let iter=0;iter<maxIter;iter++) {
        if (rho<tol2) break;
        const Ap=A.mul(p), pAp=_dotGen(p,Ap); if(Math.abs(pAp)<1e-300) break;
        const alpha=rho/pAp; const aT=A.zero.fromNumber(alpha), maT=A.zero.fromNumber(-alpha);
        const xd=x.data,rd=r.data,pd=p.data,apd=Ap.data;
        for (let i=0;i<n;i++){xd[i]=xd[i].add(aT.multiply(pd[i]));rd[i]=rd[i].add(maT.multiply(apd[i]));}
        const rhoNew=_dotGen(r,r); const beta=rhoNew/rho; rho=rhoNew;
        const bT=A.zero.fromNumber(beta);
        for (let i=0;i<n;i++) pd[i]=rd[i].add(bT.multiply(pd[i]));
    }
    return x;
}
function _dotGen<T extends INumeric<T>>(u: Matrix<T>, v: Matrix<T>): number {
    const ud=u.data,vd=v.data,n=ud.length; let s=0;
    for (let i=0;i<n;i++) s+=ud[i].conjugate().multiply(vd[i]).toNumber();
    return s;
}

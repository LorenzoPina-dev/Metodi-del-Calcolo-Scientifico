import { Matrix } from "../src";
import { performance }   from "node:perf_hooks";

for(let i=1;i<10000;i+=500){
    
    let m=Matrix.gallery.randsvd(i)
    let m2=Matrix.gallery.randsvd(i)
    let t0=performance.now()
    let m3=m.mul(m2)
    console.log(m3)
    console.log("dimensione: ", i*100)
    console.log("tempo prodotto: ",performance.now()-t0)
}

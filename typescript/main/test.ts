import { Matrix } from "../src";
import { performance } from "node:perf_hooks";
import { WorkerPool } from "../src/parallel/worker_pool";

// Abilita i worker qui!
await Matrix.initCompute({ gpu: false, workers: true });

for(let i=0; i<3000; i+=500){
    console.log("\n--- Dimensione: ", i);
    let m = Matrix.gallery.randsvd(i);
    let m2 = Matrix.gallery.randsvd(i);

    let t0 = performance.now();
    let m3 = m.mul(m2);
    console.log("Tempo prodotto sync: ", (performance.now()-t0).toFixed(2), "ms");
    
    t0 = performance.now();
    let m4 = await m.mulAsync(m2);
    console.log("Tempo prodotto async: ", (performance.now()-t0).toFixed(2), "ms");
}

// Fondamentale per terminare il processo
WorkerPool.instance.shutdown();
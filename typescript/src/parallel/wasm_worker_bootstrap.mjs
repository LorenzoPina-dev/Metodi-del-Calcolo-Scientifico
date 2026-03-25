// Bootstrap: registra tsx e carica il worker WASM TypeScript.
import { register } from "tsx/esm/api";

register();
await import("./wasm_worker.ts");

// Bootstrap: registers tsx and loads the TS matrix worker module.
import { register } from "tsx/esm/api";

register();
await import("./matrix_worker.ts");

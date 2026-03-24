// Bootstrap: registers tsx and loads the TS sync worker module.
import { register } from "tsx/esm/api";

register();
await import("./sync_worker.ts");

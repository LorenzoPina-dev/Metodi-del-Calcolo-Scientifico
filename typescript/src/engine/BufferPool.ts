// engine/BufferPool.ts - Float64Array pool to reduce GC churn

export class BufferPool {
  private static readonly MAX_PER_SIZE = 4;

  private readonly pools: Map<number, Float64Array[]> = new Map();

  private readonly supportsShared = (() => {
    try {
      return typeof SharedArrayBuffer !== "undefined";
    } catch {
      return false;
    }
  })();

  acquire(elements: number): Float64Array {
    const pool = this.pools.get(elements);
    if (pool && pool.length > 0) {
      const buf = pool.pop()!;
      buf.fill(0);
      return buf;
    }
    if (this.supportsShared) {
      try {
        const sab = new SharedArrayBuffer(elements * Float64Array.BYTES_PER_ELEMENT);
        return new Float64Array(sab);
      } catch {
        // SharedArrayBuffer not available in this context
      }
    }
    return new Float64Array(elements);
  }

  release(buf: Float64Array): void {
    const size = buf.length;
    const pool = this.pools.get(size);
    if (!pool) {
      this.pools.set(size, [buf]);
      return;
    }
    if (pool.length < BufferPool.MAX_PER_SIZE) {
      pool.push(buf);
    }
  }

  copyInto(src: ArrayLike<number>, elements: number): Float64Array {
    const out = this.acquire(elements);
    for (let i = 0; i < elements; i++) out[i] = src[i];
    return out;
  }

  flush(): void {
    this.pools.clear();
  }
}

export const globalPool = new BufferPool();

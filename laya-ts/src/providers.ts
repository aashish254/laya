/** ONNX session shim: Node (onnxruntime-node) + browser (onnxruntime-web).
 * Lazy imports only — unit tests with a fake provider never touch onnxruntime. */

/** Reviewed commit SHAs of the published checkpoints (mirror of laya/revisions.py).
 * Loads of these repos pin to the reviewed revision instead of mutable `main`; bump a
 * pin only after the new revision has been reviewed. */
export const PINNED_REVISIONS: Record<string, string> = {
  "convaiinnovations/laya": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
  "convaiinnovations/laya-multilingual": "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
  "convaiinnovations/laya-typed-decisions": "1a793eb568e6718f15941d08f85432581df534e3",
};

/** Explicit `revision` wins; published repos fall back to their reviewed pin; anything
 * else keeps the hub default (`main`). */
export function resolveRevision(repoOrId: string, revision?: string | null): string | null {
  if (revision) return revision;
  return PINNED_REVISIONS[repoOrId] ?? null;
}

/** SHA-256 hex via Web Crypto (browsers and modern Node expose globalThis.crypto). */
async function sha256Hex(data: ArrayBuffer | Uint8Array): Promise<string> {
  const subtle = (globalThis as { crypto?: { subtle?: any } }).crypto?.subtle;
  if (!subtle) {
    throw new Error("laya-ts: SHA-256 verification requires Web Crypto (globalThis.crypto.subtle)");
  }
  const digest = await subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest as ArrayBuffer), (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Reject absolute or escaping digest paths before they ever reach the filesystem. */
function normaliseDigestPath(rel: string): string {
  const norm = String(rel).replace(/\\/g, "/").replace(/^\/+/, "");
  if (!norm || norm === ".." || norm.startsWith("../") || norm.includes("/../")) {
    throw new Error(`laya-ts: unsafe path in expectedSha256: ${JSON.stringify(rel)}`);
  }
  return norm;
}

/** Verify `data` against expectedSha256[rel]; artifacts not listed are left unchecked. */
async function expectDigest(
  rel: string,
  data: ArrayBuffer | Uint8Array,
  expected: Record<string, string>,
): Promise<void> {
  const want = expected[rel] ?? expected[normaliseDigestPath(rel)];
  if (want === undefined) return;
  const got = await sha256Hex(data);
  if (got.toLowerCase() !== String(want).trim().toLowerCase()) {
    throw new Error(
      `laya-ts: SHA-256 mismatch for ${rel}: expected ${want}, got ${got}. Refusing to load the artifact.`,
    );
  }
}

export interface Batch {
  inputIds: number[][];
  attentionMask: number[][];
  markerPos: number[][];
  markerMask: boolean[][];
  qtype: number[];
}

export interface SessionProvider {
  runEncoder(batch: Batch): Promise<{ lastHidden: number[][][] }>;
  runHead(hidden: number[][][] | unknown, batch: Batch): Promise<{ logits: number[][]; act: number[][] }>;
}

function toNested(data: ArrayLike<number | bigint | boolean>, dims: number[]): any {
  const flat = Array.from(data as any, (v: any) => (typeof v === "bigint" ? Number(v) : v));
  if (dims.length === 0) return flat[0];
  const rec = (d: number, off: number): any => {
    if (d === dims.length - 1) return flat.slice(off, off + dims[d]);
    const step = dims.slice(d + 1).reduce((a, b) => a * b, 1);
    const out: any[] = [];
    for (let i = 0; i < dims[d]; i++) out.push(rec(d + 1, off + i * step));
    return out;
  };
  return rec(0, 0);
}

function i64(ort: any, arr: number[] | number[][], dims: number[]): any {
  const flat = (arr as any).flat(Infinity).map((v: number) => BigInt(Math.trunc(v)));
  return new ort.Tensor("int64", BigInt64Array.from(flat), dims);
}

/** Encoder feeds: input_ids + attention_mask (int64). */
export function feed(ort: any, b: Batch): Record<string, any> {
  const n = b.inputIds.length;
  const L = Math.max(1, ...b.inputIds.map((r) => r.length));
  return {
    input_ids: i64(ort, b.inputIds, [n, L]),
    attention_mask: i64(ort, b.attentionMask, [n, L]),
  };
}

/** Head feeds: encoder hidden + marker_pos/mask + qtype. */
export function feedHead(ort: any, hidden: number[][][] | any, b: Batch): Record<string, any> {
  const n = b.markerPos.length;
  const k = Math.max(1, ...b.markerPos.map((r) => r.length));
  const H = new ort.Tensor(
    "float32",
    Float32Array.from((hidden as any).flat(Infinity).map(Number)),
    [(hidden as any).length ?? n, (hidden as any)[0]?.length ?? 1, (hidden as any)[0]?.[0]?.length ?? 1],
  );
  const S = (hidden as any)[0]?.length ?? 1;
  // Pad/trim mask rows to S so the mask always matches hidden_states even
  // if a caller passes unpadded rows.
  const maskRows = b.attentionMask.map((r) => {
    const row = r.slice(0, S);
    while (row.length < S) row.push(0);
    return row;
  });
  return {
    hidden_states: H,
    marker_pos: i64(ort, b.markerPos, [n, k]),
    marker_mask: new ort.Tensor(
      "bool",
      Uint8Array.from((b.markerMask as any).flat(Infinity).map((v: any) => (v ? 1 : 0))),
      [n, k],
    ),
    qtype: i64(ort, b.qtype.map((v) => [v]), [n, 1]),
    // Padding mask for the head transformer (py DecisionModel.forward).
    // Without it, batch mates of unequal length corrupt each other's markers.
    attention_mask: i64(ort, maskRows, [n, S]),
  };
}

function pickOutput(out: Record<string, any>, names: string[]): any {
  for (const n of names) if (out[n] !== undefined) return out[n];
  const vals = Object.values(out);
  return vals[0];
}

export interface ProviderOptions {
  device?: string;
  numThreads?: number;
  /** Opt-in {artifact name: SHA-256 hexdigest} check for fetched ONNX files (web). */
  expectedSha256?: Record<string, string>;
}

function applyNumThreads(ort: any, numThreads?: number): void {
  try {
    const raw =
      numThreads ??
      (typeof process !== "undefined" ? Number((process as any).env?.["LAYA_THREADS"]) : NaN);
    if (Number.isFinite(raw) && (raw as number) > 0 && ort?.env) {
      ort.env.numThreads = Math.trunc(raw as number);
    }
  } catch {
    /* best-effort only */
  }
}

function isOomError(e: unknown): boolean {
  const m = String((e as any)?.message ?? e).toLowerCase();
  return m.includes("memory") || m.includes("cuda") || m.includes("out of memory") || m.includes("oom");
}

/** Online-first fetch: try network, cache on success, fall back to CacheStorage. */
async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  const g = globalThis as unknown as { caches?: any };
  let cache: any = null;
  let hit: any = null;
  try {
    if (g.caches && typeof g.caches.open === "function") {
      try {
        cache = await g.caches.open("laya-ts");
        try {
          hit = await cache.match(url);
        } catch {
          hit = null;
        }
      } catch {
        cache = null;
      }
    }
  } catch {
    cache = null;
  }
  if (cache) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        try {
          await cache.put(url, res.clone());
        } catch {
          /* cache full/blocked: still return network bytes */
        }
        return await res.arrayBuffer();
      }
    } catch (e) {
      if (hit) {
        try {
          return await hit.arrayBuffer();
        } catch {
          /* fall through to throw original */
        }
      }
      throw e;
    }
    if (hit) {
      try {
        return await hit.arrayBuffer();
      } catch {
        /* fall through to direct error below */
      }
    }
    throw new Error(`fetch failed for ${url}`);
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch failed for ${url}: ${res.status}`);
  return await res.arrayBuffer();
}

export interface NodeBundle {
  dir: string;
  cfg: any;
  tokenizerJson: unknown | null;
  /** Commit SHA the artifacts came from (pinned/requested, or the hub's `x-repo-commit`); null for local dirs. */
  revision: string | null;
}

export async function loadNodeBundle(
  modelDirOrRepo: string,
  opts?: {
    subfolder?: string | null;
    localDir?: string;
    token?: string | null;
    revision?: string | null;
    expectedSha256?: Record<string, string>;
  },
): Promise<NodeBundle> {
  const fs: typeof import("node:fs/promises") = await import("node:fs/promises");
  const path: typeof import("node:path") = await import("node:path");
  const os: typeof import("node:os") = await import("node:os");
  const sub = opts?.subfolder ?? null;
  let dir = opts?.localDir ?? modelDirOrRepo;
  let resolvedRevision: string | null = null;
  try {
    const st = await fs.stat(sub ? path.join(dir, sub) : dir);
    if (st.isDirectory()) dir = sub ? path.join(dir, sub) : dir;
    else dir = path.dirname(dir);
  } catch {
    // Pin published repos to their reviewed commit SHA instead of mutable `main`; the
    // revision joins the cache key so differently-pinned artifacts never collide.
    const revision = resolveRevision(modelDirOrRepo, opts?.revision);
    resolvedRevision = revision;
    const cache = path.join(
      os.homedir(),
      ".cache",
      "laya-ts",
      "hf",
      modelDirOrRepo.replace(/\//g, "__"),
      sub ?? "root",
      ...(revision && revision !== "main" ? [revision] : []),
    );
    await fs.mkdir(cache, { recursive: true });
    const token =
      opts?.token ?? (typeof process !== "undefined" ? (process as any).env?.["HF_TOKEN"] : undefined);
    for (const f of ["rl_agent_config.json", "tokenizer.json", "tokenizer/tokenizer.json", "encoder.onnx", "head.onnx"]) {
      try {
        await fs.stat(path.join(cache, f));
      } catch {
        const url = `https://huggingface.co/${modelDirOrRepo}/resolve/${revision ?? "main"}/${sub ? sub + "/" : ""}${f}`;
        const res = await fetch(url, token ? { headers: { Authorization: `Bearer ${token}` } } : undefined);
        const commit = res.headers?.get?.("x-repo-commit");
        if (commit) resolvedRevision = commit;
        if (!res.ok) {
          if (f === "rl_agent_config.json") {
            throw new Error(
              `Incompatible model: ${JSON.stringify(modelDirOrRepo)} does not contain 'rl_agent_config.json'.`,
            );
          }
          continue;
        }
        const target = path.join(cache, f);
        await fs.mkdir(path.dirname(target), { recursive: true });
        // Write-then-rename so an interrupted download never leaves a truncated
        // artifact that later loads treat as complete.
        const tmp = `${target}.tmp-${typeof process !== "undefined" ? process.pid : 0}`;
        await fs.writeFile(tmp, new Uint8Array(await res.arrayBuffer()));
        await fs.rename(tmp, target);
      }
    }
    dir = cache;
  }
  // Opt-in integrity check over the resolved directory (covers local dirs, warm cache,
  // and fresh downloads alike) before any artifact is parsed or executed.
  if (opts?.expectedSha256) {
    const { createHash } = await import("node:crypto");
    for (const [rel, want] of Object.entries(opts.expectedSha256)) {
      const norm = normaliseDigestPath(rel);
      let buf: Uint8Array;
      try {
        buf = new Uint8Array(await fs.readFile(path.join(dir, norm)));
      } catch {
        throw new Error(`laya-ts: cannot verify ${JSON.stringify(rel)}: not found under ${dir}`);
      }
      const got = createHash("sha256").update(buf).digest("hex");
      if (got.toLowerCase() !== String(want).trim().toLowerCase()) {
        throw new Error(
          `laya-ts: SHA-256 mismatch for ${rel}: expected ${want}, got ${got}. Refusing to load the artifact.`,
        );
      }
    }
  }
  let cfg: any = {};
  try {
    cfg = JSON.parse(await fs.readFile(path.join(dir, "rl_agent_config.json"), "utf8"));
  } catch {
    throw new Error(
      `Incompatible model: ${JSON.stringify(modelDirOrRepo)} does not contain 'rl_agent_config.json'.`,
    );
  }
  let tokenizerJson: unknown | null = null;
  for (const candidate of ["tokenizer.json", "tokenizer/tokenizer.json"]) {
    try {
      tokenizerJson = JSON.parse(await fs.readFile(path.join(dir, candidate), "utf8"));
      break;
    } catch {
      // Try the next supported Hugging Face layout.
    }
  }
  return { dir, cfg, tokenizerJson, revision: resolvedRevision };
}

export interface WebBundle {
  dir: string;
  cfg: any;
  tokenizerJson: unknown | null;
  /** Pinned/requested commit SHA, if any (full-URL sources have no implicit revision). */
  revision: string | null;
}

function baseUrlFor(repoOrUrl: string, subfolder?: string | null, revision?: string | null): string {
  const sub = subfolder ? `/${subfolder.replace(/^\/+|\/+$/g, "")}` : "";
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(repoOrUrl)) {
    return `${repoOrUrl.replace(/\/+$/, "")}${sub}`;
  }
  return `https://huggingface.co/${repoOrUrl}/resolve/${revision ?? "main"}${sub}`;
}

export async function loadWebBundle(
  repoOrUrl: string,
  opts?: {
    subfolder?: string | null;
    revision?: string | null;
    expectedSha256?: Record<string, string>;
  },
): Promise<WebBundle> {
  const revision = resolveRevision(repoOrUrl, opts?.revision);
  const base = baseUrlFor(repoOrUrl, opts?.subfolder ?? null, revision);
  const fetchVerifiedJson = async (rel: string): Promise<unknown> => {
    const buf = await fetchArrayBuffer(`${base}/${rel}`);
    if (opts?.expectedSha256) await expectDigest(rel, buf, opts.expectedSha256);
    return JSON.parse(new TextDecoder().decode(buf));
  };
  let cfg: any;
  try {
    cfg = await fetchVerifiedJson("rl_agent_config.json");
  } catch (e) {
    if (e instanceof Error && e.message.startsWith("laya-ts: SHA-256 mismatch")) throw e;
    throw new Error(`Incompatible model: ${JSON.stringify(repoOrUrl)} does not contain 'rl_agent_config.json'.`);
  }
  let tokenizerJson: unknown | null = null;
  for (const candidate of ["tokenizer.json", "tokenizer/tokenizer.json"]) {
    try {
      tokenizerJson = await fetchVerifiedJson(candidate);
      break;
    } catch (e) {
      if (e instanceof Error && e.message.startsWith("laya-ts: SHA-256 mismatch")) throw e;
      // Try the next supported Hugging Face layout.
    }
  }
  return { dir: base, cfg, tokenizerJson, revision };
}

export async function createNodeProvider(
  modelDir: string,
  opts?: ProviderOptions,
): Promise<SessionProvider> {
  const spec = "onnxruntime-" + "node";
  const ort: any = await import(/* @vite-ignore */ spec);
  applyNumThreads(ort, opts?.numThreads);
  try {
    const fs: typeof import("node:fs/promises") = await import("node:fs/promises");
    const path: typeof import("node:path") = await import("node:path");
    for (const f of ["encoder.onnx", "head.onnx"]) {
      const p = path.join(modelDir, f);
      try {
        await fs.stat(p);
      } catch {
        throw new Error(`Incompatible model: '${f}' not found in ${JSON.stringify(modelDir)} (expected ${p}).`);
      }
    }
  } catch (e) {
    if (e instanceof Error && e.message.includes("not found in")) throw e;
  }
  const dev = String(opts?.device ?? "cpu").toLowerCase();
  const want = dev === "cuda" ? "cuda" : dev === "dml" ? "dml" : "cpu";
  const make = async (ep: string) => {
    const e = await ort.InferenceSession.create(`${modelDir}/encoder.onnx`, {
      executionProviders: [ep],
    });
    const h = await ort.InferenceSession.create(`${modelDir}/head.onnx`, {
      executionProviders: ["cpu"],
    });
    return { e, h };
  };
  let enc: any;
  let head: any;
  let activeEP = want;
  try {
    ({ e: enc, h: head } = await make(want));
  } catch (e) {
    if (want !== "cpu") {
      console.warn(`Warning: ${want.toUpperCase()} requested but not available. Falling back to CPU.`);
      ({ e: enc, h: head } = await make("cpu"));
      activeEP = "cpu";
    } else {
      throw e;
    }
  }
  let cpuEnc: any = null;
  let cpuHead: any = null;
  const ensureCpu = async () => {
    if (!cpuEnc) {
      cpuEnc = await ort.InferenceSession.create(`${modelDir}/encoder.onnx`, {
        executionProviders: ["cpu"],
      });
      cpuHead = await ort.InferenceSession.create(`${modelDir}/head.onnx`, {
        executionProviders: ["cpu"],
      });
    }
    return { cpuEnc, cpuHead };
  };
  const runWithCpuFallback = async <T>(fn: (e: any, h: any) => Promise<T>): Promise<T> => {
    try {
      return await fn(enc, head);
    } catch (e) {
      if (activeEP !== "cpu" && isOomError(e)) {
        console.warn("Warning: GPU memory exceeded during inference. Falling back to CPU...");
        const { cpuEnc: ce, cpuHead: ch } = await ensureCpu();
        enc = ce;
        head = ch;
        activeEP = "cpu";
        return await fn(enc, head);
      }
      if (isOomError(e)) {
        throw new Error(`${(e as Error).message} (GPU out of memory; try device: "cpu")`);
      }
      throw e;
    }
  };
  return {
    runEncoder: async (b) =>
      runWithCpuFallback(async (e) => {
        const out = await e.run(feed(ort, b));
        const t = pickOutput(out, ["last_hidden_state", "lastHidden", "hidden_states"]);
        return { lastHidden: toNested(t.data, t.dims) };
      }),
    runHead: async (h, b) =>
      runWithCpuFallback(async (_e, hd) => {
        const out = await hd.run(feedHead(ort, h, b));
        const vals = Object.values(out) as any[];
        const lt = pickOutput(out, ["logits"]);
        const at = pickOutput(out, ["act_logits", "act"]) ?? vals[1] ?? vals[0];
        return { logits: toNested(lt.data, lt.dims), act: toNested(at.data, at.dims) };
      }),
  };
}

export async function createWebProvider(
  modelUrl: string,
  opts?: ProviderOptions,
): Promise<SessionProvider> {
  const spec = "onnxruntime-" + "web";
  const ort: any = await import(/* @vite-ignore */ spec);
  applyNumThreads(ort, opts?.numThreads);
  const base = modelUrl.replace(/\/+$/, "");
  const encUrl = `${base}/encoder.onnx`;
  const headUrl = `${base}/head.onnx`;
  let encBuf: ArrayBuffer;
  try {
    encBuf = await fetchArrayBuffer(encUrl);
  } catch {
    throw new Error(`Incompatible model: 'encoder.onnx' not found (expected ${encUrl}).`);
  }
  let headBuf: ArrayBuffer;
  try {
    headBuf = await fetchArrayBuffer(headUrl);
  } catch {
    throw new Error(`Incompatible model: 'head.onnx' not found (expected ${headUrl}).`);
  }
  // Verify before the bytes reach the runtime: a tampered ONNX never becomes a session.
  if (opts?.expectedSha256) {
    await expectDigest("encoder.onnx", encBuf, opts.expectedSha256);
    await expectDigest("head.onnx", headBuf, opts.expectedSha256);
  }
  let enc: any;
  try {
    enc = await ort.InferenceSession.create(new Uint8Array(encBuf), {
      executionProviders: ["webgpu", "wasm"],
    });
  } catch (e) {
    enc = await ort.InferenceSession.create(new Uint8Array(encBuf), {
      executionProviders: ["wasm"],
    });
  }
  const head = await ort.InferenceSession.create(new Uint8Array(headBuf), {
    executionProviders: ["wasm"],
  });
  return {
    runEncoder: async (b) => {
      try {
        const out = await enc.run(feed(ort, b));
        const t = pickOutput(out, ["last_hidden_state", "lastHidden", "hidden_states"]);
        return { lastHidden: toNested(t.data, t.dims) };
      } catch (e) {
        if (isOomError(e)) throw new Error(`${(e as Error).message} (WebGPU out of memory; WASM fallback already active)`);
        throw e;
      }
    },
    runHead: async (h, b) => {
      try {
        const out = await head.run(feedHead(ort, h, b));
        const vals = Object.values(out) as any[];
        const lt = pickOutput(out, ["logits"]);
        const at = pickOutput(out, ["act_logits", "act"]) ?? vals[1] ?? vals[0];
        return { logits: toNested(lt.data, lt.dims), act: toNested(at.data, at.dims) };
      } catch (e) {
        if (isOomError(e)) throw new Error(`${(e as Error).message} (out of memory; try fewer questions per call)`);
        throw e;
      }
    },
  };
}

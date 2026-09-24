import { describe, expect, it } from "vitest";
import { Agent } from "../src/agent.js";
const fakeProvider = () => ({
  async runEncoder(_b: any) { return { lastHidden: [[1, 0], [0, 1]] }; },
  async runHead(_h: any) { return { logits: [[2, 0]], act: [[3, 0]] }; },
});
describe("agent", () => {
  it("empty questions skip forward pass", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    expect(await a.systemOne("hi", {})).toEqual(
      expect.objectContaining({ answers: {} }));
  });
  it("choice picks argmax with temp + confidence", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    const r: any = await a.systemOne("hi", {
      d: { type: "choice", instructions: "q?", criteria: { x: "yes", y: "no" } } });
    expect(r.answers.d.choice).toBe("x");
    expect(r.usage.input_tokens).toBeGreaterThan(0);
  });
  it("rejects bad question with qid", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    await expect(a.systemOne("hi", { q: { type: "nope" } as any })).rejects.toThrow('question "q"');
  });
});

describe("minConfidence (#361)", () => {
  const q = { d: { type: "choice", instructions: "q?", criteria: { x: "yes", y: "no" } } };

  it("flags answers below the threshold, answer and confidence intact", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    const r: any = await a.systemOne("hi", q as any, { minConfidence: 0.99 });
    expect(r.answers.d.low_confidence).toBe(true);
    expect(r.answers.d.choice).toBe("x");
    expect(typeof r.answers.d.confidence).toBe("number");
  });

  it("leaves answers at or above the threshold unflagged", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    const r: any = await a.systemOne("hi", q as any, { minConfidence: 0.4 });
    expect(r.answers.d.low_confidence).toBeUndefined();
    expect(r.answers.d.choice).toBe("x");
  });

  it("defaults to off and rejects out-of-range values", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    const r: any = await a.systemOne("hi", q as any);
    expect(r.answers.d.low_confidence).toBeUndefined();
    await expect(a.systemOne("hi", q as any, { minConfidence: 1.5 })).rejects.toThrow("minConfidence");
    await expect(a.systemOne("hi", q as any, { minConfidence: -0.1 })).rejects.toThrow("minConfidence");
  });

  it("flags hook-provided results that carry a confidence", async () => {
    const a = new Agent({ provider: fakeProvider() } as any);
    const r: any = await a.systemOne("hi", q as any, {
      minConfidence: 0.9,
      onPredictStart: (_ctx: any) => {},
      onPredictEnd: (ctx: any) => {
        ctx.results = [{ model: "laya-rl-agent", answers: { d: { type: "choice", choice: "y", confidence: 0.3 } }, usage: { input_tokens: 0, output_tokens: 0 } }];
      },
    });
    expect(r.answers.d.low_confidence).toBe(true);
    expect(r.answers.d.choice).toBe("y");
  });
});

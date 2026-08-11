import { beforeEach, describe, expect, it, vi } from "vitest";
import type { PineconeRecord } from "@pinecone-database/pinecone";

// Mock the transformers pipeline so the batching logic can be tested offline,
// without downloading a model or running inference. The mock returns a fixed
// 3-dim embedding regardless of input.
const MOCK_EMBEDDING = [0.1, 0.2, 0.3];
vi.mock("@huggingface/transformers", () => ({
  pipeline: vi.fn(async () => async (_text: string) => ({
    data: Float32Array.from(MOCK_EMBEDDING),
  })),
  AutoConfig: { from_pretrained: vi.fn(async () => ({})) },
}));

// Imported after the mock is registered.
const { embedder } = await import("../../src/embeddings.js");
const { Document } = await import("@langchain/core/documents");

describe("Embedder.embedBatch", () => {
  beforeEach(async () => {
    await embedder.init("mock-model");
  });

  it("splits documents into batches of the requested size", async () => {
    const documents = ["a", "b", "c", "d", "e"];
    const batches: PineconeRecord[][] = [];

    await embedder.embedBatch(documents, 2, (embeddings) => {
      batches.push(embeddings);
    });

    // 5 documents, batch size 2 -> [2, 2, 1]
    expect(batches.map((b) => b.length)).toEqual([2, 2, 1]);
  });

  it("produces one embedding per document with the model's vector values", async () => {
    const documents = ["only-one"];
    const collected: PineconeRecord[] = [];

    await embedder.embedBatch(documents, 5, (embeddings) => {
      collected.push(...embeddings);
    });

    expect(collected).toHaveLength(1);
    // Values round-trip through a Float32Array, so compare with tolerance
    // rather than for exact float equality.
    expect(collected[0].values).toHaveLength(MOCK_EMBEDDING.length);
    collected[0].values.forEach((v, i) =>
      expect(v).toBeCloseTo(MOCK_EMBEDDING[i], 5)
    );
    // A string document with no metadata falls back to storing its own text.
    expect(collected[0].metadata).toEqual({ text: "only-one" });
    expect(typeof collected[0].id).toBe("string");
  });

  it("preserves document metadata and uses its id when embedding Documents", async () => {
    const documents = [
      new Document({ pageContent: "context text", metadata: { id: "doc-42" } }),
    ];
    const collected: PineconeRecord[] = [];

    await embedder.embedBatch(documents, 1, (embeddings) => {
      collected.push(...embeddings);
    });

    expect(collected[0].id).toBe("doc-42");
    expect(collected[0].metadata).toEqual({ id: "doc-42" });
  });
});

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { Pinecone, type PineconeRecord } from "@pinecone-database/pinecone";
import { embedder } from "../../src/embeddings.js";

// Live end-to-end test: exercises a real embed -> upsert -> query round-trip
// against Pinecone. It is GATED and SELF-SKIPPING — with no credentials it
// skips cleanly (so `npm test` is green locally and on untrusted-PR CI), and it
// only runs on main / workflow_dispatch where secrets are available.
//
// It gates on PINECONE_API_KEY alone: the serverless v8 SDK derives the region
// from the index, so there is no PINECONE_ENVIRONMENT. Ingestion + retrieval
// need only Pinecone and the local transformers embedder — no OpenAI.
const apiKey = process.env.PINECONE_API_KEY;
const hasCredentials = Boolean(apiKey);

// all-MiniLM-L6-v2 produces 384-dim vectors.
const MODEL = "Xenova/all-MiniLM-L6-v2";
const DIMENSION = 384;
const NAMESPACE = "default";

// A uniquely-named throwaway index so concurrent runs never collide and cleanup
// can never clobber a real index. Index names are lowercase alphanumeric+hyphen.
const indexName = `e2e-ingest-retrieve-${Date.now().toString(36)}`;

describe.skipIf(!hasCredentials)("live ingestion + retrieval", () => {
  // Constructed in beforeAll, not at collection time: the v8 client validates
  // its config in the constructor, so `new Pinecone({ apiKey: undefined })`
  // would throw while collecting even when this suite is skipped for lack of
  // credentials. beforeAll runs only when the suite actually executes.
  let pinecone: Pinecone;

  beforeAll(() => {
    pinecone = new Pinecone({ apiKey: apiKey! });
  });

  afterAll(async () => {
    // Tear down the throwaway index even if an assertion failed mid-test.
    try {
      await pinecone.deleteIndex(indexName);
    } catch (error) {
      console.error(`Failed to delete test index ${indexName}:`, error);
    }
  });

  it("upserts embedded documents and retrieves the most relevant one", async () => {
    // Distinct facts so retrieval has an unambiguous best match to return.
    const documents = [
      { id: "sky", text: "The sky appears blue because of Rayleigh scattering." },
      { id: "grass", text: "Grass is green because it contains chlorophyll." },
      { id: "sun", text: "The sun is a star at the center of the solar system." },
    ];

    // `waitUntilReady` blocks until the index can accept upserts;
    // `suppressConflicts` makes re-runs idempotent.
    await pinecone.createIndex({
      name: indexName,
      dimension: DIMENSION,
      metric: "cosine",
      spec: { serverless: { cloud: "aws", region: "us-east-1" } },
      waitUntilReady: true,
      suppressConflicts: true,
    });

    const index = pinecone.index({ name: indexName }).namespace(NAMESPACE);

    await embedder.init(MODEL);
    const vectors: PineconeRecord[] = [];
    await embedder.embedBatch(
      documents.map((d) => d.text),
      documents.length,
      (embeddings) => {
        // Re-key each vector by its stable document id so we can assert which
        // one comes back (embedBatch assigns random UUIDs to bare strings).
        embeddings.forEach((v, i) => {
          v.id = documents[i].id;
          v.metadata = { text: documents[i].text };
        });
        vectors.push(...embeddings);
      },
    );

    await index.upsert({ records: vectors });

    // Upserts are eventually consistent — poll describeIndexStats with backoff
    // until all vectors are visible, instead of a single fixed sleep.
    await waitForVectorCount(index, documents.length);

    const [queryVector] = await embedQuery("Why does the sky look blue during the day?");
    const result = await index.query({
      topK: 1,
      vector: queryVector,
      includeMetadata: true,
    });

    expect(result.matches?.[0]?.id).toBe("sky");
  });

  // Embed a single query string to a raw vector for querying.
  async function embedQuery(text: string): Promise<number[][]> {
    const out: number[][] = [];
    await embedder.embedBatch([text], 1, (embeddings) => {
      out.push(embeddings[0].values as number[]);
    });
    return out;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async function waitForVectorCount(index: any, expected: number): Promise<void> {
    const maxAttempts = 20;
    let delayMs = 1000;
    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
      const stats = await index.describeIndexStats();
      const count = stats.namespaces?.[NAMESPACE]?.recordCount ?? 0;
      if (count >= expected) return;
      await new Promise((resolve) => setTimeout(resolve, delayMs));
      delayMs = Math.min(delayMs * 2, 8000); // exponential backoff, capped
    }
    throw new Error(
      `Only ${expected} vectors expected but never became visible in index ${indexName}`,
    );
  }
});

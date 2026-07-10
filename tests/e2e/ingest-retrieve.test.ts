import { afterAll, describe, expect, it } from "vitest";
import { PineconeClient, utils, type Vector } from "@pinecone-database/pinecone";
import { embedder } from "../../src/embeddings.js";

const { createIndexIfNotExists, chunkedUpsert, waitUntilIndexIsReady } = utils;

// Live end-to-end test: exercises a real embed -> upsert -> query round-trip
// against Pinecone. It is GATED and SELF-SKIPPING — with no credentials it
// skips cleanly (so `npm test` is green locally and on untrusted-PR CI), and it
// only runs on main / workflow_dispatch where secrets are available.
//
// It gates on PINECONE_API_KEY (+ PINECONE_ENVIRONMENT, required by the current
// pod-based SDK). Ingestion + retrieval need only Pinecone and the local
// transformers embedder — no OpenAI. When #16 modernizes the stack this file is
// re-pointed at the serverless v8 API (dropping PINECONE_ENVIRONMENT), and an
// agent-level e2e (which needs OPENAI_API_KEY) becomes possible once chat.ts is
// importable. See issue #16.
const apiKey = process.env.PINECONE_API_KEY;
const environment = process.env.PINECONE_ENVIRONMENT;
const hasCredentials = Boolean(apiKey && environment);

// all-MiniLM-L6-v2 produces 384-dim vectors.
const MODEL = "Xenova/all-MiniLM-L6-v2";
const DIMENSION = 384;
const NAMESPACE = "default";

// A uniquely-named throwaway index so concurrent runs never collide and cleanup
// can never clobber a real index. Index names are lowercase alphanumeric+hyphen.
const indexName = `e2e-ingest-retrieve-${Date.now().toString(36)}`;

describe.skipIf(!hasCredentials)("live ingestion + retrieval", () => {
  const client = new PineconeClient();

  afterAll(async () => {
    // Tear down the throwaway index even if an assertion failed mid-test.
    try {
      await client.deleteIndex({ indexName });
    } catch (error) {
      console.error(`Failed to delete test index ${indexName}:`, error);
    }
  });

  it("upserts embedded documents and retrieves the most relevant one", async () => {
    await client.init({ apiKey: apiKey!, environment: environment! });

    // Distinct facts so retrieval has an unambiguous best match to return.
    const documents = [
      { id: "sky", text: "The sky appears blue because of Rayleigh scattering." },
      { id: "grass", text: "Grass is green because it contains chlorophyll." },
      { id: "sun", text: "The sun is a star at the center of the solar system." },
    ];

    await client.createIndex({
      createRequest: { name: indexName, dimension: DIMENSION, metric: "cosine" },
    });
    // Poll for readiness rather than sleeping a fixed interval — index creation
    // latency is variable.
    await waitUntilIndexIsReady(client, indexName);
    await createIndexIfNotExists(client, indexName, DIMENSION);

    const index = client.Index(indexName);

    await embedder.init(MODEL);
    const vectors: Vector[] = [];
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

    await chunkedUpsert(index, vectors, NAMESPACE);

    // Upserts are eventually consistent — poll describeIndexStats with backoff
    // until all vectors are visible, instead of a single fixed sleep.
    await waitForVectorCount(index, documents.length);

    const [queryVector] = (
      await embedQuery("Why does the sky look blue during the day?")
    );
    const result = await index.query({
      queryRequest: {
        vector: queryVector,
        topK: 1,
        namespace: NAMESPACE,
        includeMetadata: true,
      },
    });

    expect(result.matches?.[0]?.id).toBe("sky");
  });

  // Embed a single query string to a raw vector for querying.
  async function embedQuery(text: string): Promise<number[][]> {
    const out: number[][] = [];
    await embedder.embedBatch([text], 1, (embeddings) => {
      out.push(embeddings[0].values);
    });
    return out;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async function waitForVectorCount(index: any, expected: number): Promise<void> {
    const maxAttempts = 20;
    let delayMs = 1000;
    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
      const stats = await index.describeIndexStats({ describeIndexStatsRequest: {} });
      const count = stats.namespaces?.[NAMESPACE]?.vectorCount ?? 0;
      if (count >= expected) return;
      await new Promise((resolve) => setTimeout(resolve, delayMs));
      delayMs = Math.min(delayMs * 2, 8000); // exponential backoff, capped
    }
    throw new Error(
      `Only ${expected} vectors expected but never became visible in index ${indexName}`,
    );
  }
});

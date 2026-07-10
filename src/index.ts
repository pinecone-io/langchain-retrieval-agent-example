/* eslint-disable import/no-extraneous-dependencies */
/* eslint-disable dot-notation */
import * as dotenv from "dotenv";
import { PineconeRecord } from "@pinecone-database/pinecone";
import { getEnv } from "utils/util.ts";
import { getPineconeClient } from "utils/pinecone.ts";
import cliProgress from "cli-progress";
import { Document } from "@langchain/core/documents";
import * as dfd from "danfojs-node";
import { embedder } from "embeddings.ts";
import { SquadRecord, loadSquad } from "./utils/squadLoader.js";

dotenv.config();

// all-MiniLM-L6-v2 produces 384-dimensional embeddings.
const EMBEDDING_DIMENSION = 384;
// Embed and upsert in batches rather than one network round-trip per record
// (the README documents a batch size of 100).
const UPSERT_BATCH_SIZE = 100;
const NAMESPACE = "default";

const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);

// Index setup
const indexName = getEnv("PINECONE_INDEX");
const pinecone = getPineconeClient();


async function getChunk(df: dfd.DataFrame, start: number, size: number): Promise<dfd.DataFrame> {
  // eslint-disable-next-line no-return-await
  return await df.head(start + size).tail(size);
}

async function* processInChunks(dataFrame: dfd.DataFrame, chunkSize: number): AsyncGenerator<Document[]> {
  for (let i = 0; i < dataFrame.shape[0]; i += chunkSize) {
    const chunk = await getChunk(dataFrame, i, chunkSize);
    const records = dfd.toJSON(chunk) as SquadRecord[];
    yield records.map((record: SquadRecord) => new Document({
      pageContent: record.context,
      metadata: {
        id: record["id"],
        question: record["question"],
        answer: record["answer"],
        context: record["context"],
      },
    }));
  }
}

async function embedAndUpsert(dataFrame: dfd.DataFrame, chunkSize: number) {
  const chunkGenerator = processInChunks(dataFrame, chunkSize);
  const index = pinecone.index({ name: indexName }).namespace(NAMESPACE);

  for await (const documents of chunkGenerator) {
    await embedder.embedBatch(documents, chunkSize, async (embeddings: PineconeRecord[]) => {
      await index.upsert({ records: embeddings });
      progressBar.increment(embeddings.length);
    });
  }
}

try {
  const squadData = await loadSquad();
  // squadData.print();
  // Idempotent: `suppressConflicts` makes this a no-op if the index already
  // exists, and `waitUntilReady` blocks until it can accept upserts.
  await pinecone.createIndex({
    name: indexName,
    dimension: EMBEDDING_DIMENSION,
    metric: "cosine",
    spec: { serverless: { cloud: "aws", region: "us-east-1" } },
    waitUntilReady: true,
    suppressConflicts: true,
  });
  progressBar.start(squadData.shape[0], 0);
  await embedder.init("Xenova/all-MiniLM-L6-v2");
  await embedAndUpsert(squadData, UPSERT_BATCH_SIZE);

  progressBar.stop();
  console.log(`Inserted ${progressBar.getTotal()} documents into index ${indexName}`);

} catch (error) {
  console.error(error);
}

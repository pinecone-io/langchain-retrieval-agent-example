import { randomUUID } from "crypto";
import { Pipeline, pipeline } from "@xenova/transformers";
import { Vector } from "@pinecone-database/pinecone";
import { sliceIntoChunks } from "./utils/util.js";


class Embedder {
  private pipe: Pipeline;

  async init() {
    this.pipe = await pipeline(
      "embeddings",
      "Xenova/all-MiniLM-L6-v2"
    );
  }

  // Embeds a text and returns the embedding
  async embed(text: string): Promise<Vector> {
    const result = await this.pipe(text);
    return {
      id: randomUUID(),
      metadata: {
        text,
      },
      values: Array.from(result.data),
    };
  }

  // Embeds a batch of texts and calls onDoneBatch with the embeddings
  async embedBatch(
    texts: string[],
    batchSize: number,
    onDoneBatch: (embeddings: Vector[]) => void
  ) {
    const batches = sliceIntoChunks<string>(texts, batchSize);
    for (const batch of batches) {
      const embeddings = await Promise.all(
        batch.map((text) => this.embed(text))
      );
      await onDoneBatch(embeddings);
    }
  }
}

const embedder = new Embedder();

export { embedder };

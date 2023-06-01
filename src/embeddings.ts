import { randomUUID } from "crypto";
import { Pipeline, pipeline } from "@xenova/transformers";
import { Vector } from "@pinecone-database/pinecone";
import { Document } from 'langchain/document';
import { EmbeddingsParams, Embeddings } from "langchain/embeddings/base";
import { sliceIntoChunks } from "./utils/util.js";

type DocumentOrString = Document | string;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isString(test: any): test is string {
  return typeof test === "string";
}

class Embedder {
  private pipe: Pipeline;

  async init(modelName: string) {
    this.pipe = await pipeline(
      "embeddings",
      modelName
    );
  }

  // Embeds a text and returns the embedding
  async embed(text: string, metadata?: Record<string, unknown>): Promise<Vector> {
    const result = await this.pipe(text);
    const id = (metadata?.id as string) || randomUUID();

    return {
      id,
      metadata: metadata || {
        text,
      },
      values: Array.from(result.data),
    };
  }

  // Embeds a batch of documents and calls onDoneBatch with the embeddings
  async embedBatch(
    documents: DocumentOrString[],
    batchSize: number,
    onDoneBatch: (embeddings: Vector[]) => void
  ) {
    const batches = sliceIntoChunks<DocumentOrString>(documents, batchSize);
    for (const batch of batches) {
      const embeddings = await Promise.all(
        batch.map((documentOrString) =>
          isString(documentOrString)
            ? this.embed(documentOrString)
            : this.embed(documentOrString.pageContent, documentOrString.metadata)
        )
      );
      await onDoneBatch(embeddings);
    }
  }
}

interface TransformersJSEmbeddingParams extends EmbeddingsParams {
  modelName: string;
  onEmbeddingDone?: (embeddings: Vector[]) => void;
}

class TransformersJSEmbedding extends Embeddings implements TransformersJSEmbeddingParams {
  modelName: string;

  pipe: Pipeline | null = null;

  constructor(params: TransformersJSEmbeddingParams) {
    super(params);
    this.modelName = params.modelName;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    this.pipe = this.pipe || await pipeline(
      "embeddings",
      this.modelName
    );

    const embeddings = await Promise.all(texts.map(async (text) => this.embedQuery(text)));
    return embeddings;
  }

  async embedQuery(text: string): Promise<number[]> {
    this.pipe = this.pipe || await pipeline(
      "embeddings",
      this.modelName
    );

    const result = await this.pipe(text);
    return Array.from(result.data) as number[];
  }
}


const embedder = new Embedder();
export { embedder, TransformersJSEmbedding };

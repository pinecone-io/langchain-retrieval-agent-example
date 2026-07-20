import { randomUUID } from "crypto";
import { FeatureExtractionPipeline, pipeline, AutoConfig } from "@huggingface/transformers";
import { PineconeRecord, RecordMetadata } from "@pinecone-database/pinecone";
import { Document } from "@langchain/core/documents";
import { EmbeddingsParams, Embeddings } from "@langchain/core/embeddings";
import { sliceIntoChunks } from "./utils/util.js";

// all-MiniLM-L6-v2 needs mean pooling over tokens + L2 normalization to collapse
// a text into a single sentence embedding. Without pooling the pipeline returns
// the full [tokens x 384] tensor, which flattens to a wrong-sized vector that the
// index rejects ("Vector dimension N does not match ... 384" — see issue #3).
const POOLING_OPTIONS = { pooling: "mean", normalize: true } as const;

type DocumentOrString = Document | string;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isString(test: any): test is string {
  return typeof test === "string";
}

class Embedder {
  private pipe: FeatureExtractionPipeline;

  async init(modelName: string) {
    const config = await AutoConfig.from_pretrained(modelName);
    this.pipe = await pipeline(
      "feature-extraction",
      modelName,
      { dtype: "fp32", config }
    );
  }

  // Embeds a text and returns the embedding
  async embed(text: string, metadata?: Record<string, unknown>): Promise<PineconeRecord> {
    const result = await this.pipe(text, POOLING_OPTIONS);
    const id = (metadata?.id as string) || randomUUID();

    return {
      id,
      metadata: (metadata || { text }) as RecordMetadata,
      values: Array.from(result.data as Float32Array),
    };
  }

  // Embeds a batch of documents and calls onDoneBatch with the embeddings
  async embedBatch(
    documents: DocumentOrString[],
    batchSize: number,
    onDoneBatch: (embeddings: PineconeRecord[]) => void
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
  onEmbeddingDone?: (embeddings: PineconeRecord[]) => void;
}

class TransformersJSEmbedding extends Embeddings implements TransformersJSEmbeddingParams {
  modelName: string;

  pipe: FeatureExtractionPipeline | null = null;

  constructor(params: TransformersJSEmbeddingParams) {
    super(params);
    this.modelName = params.modelName;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    this.pipe = this.pipe || await pipeline(
      "feature-extraction",
      this.modelName
    );

    const embeddings = await Promise.all(texts.map(async (text) => this.embedQuery(text)));
    return embeddings;
  }

  async embedQuery(text: string): Promise<number[]> {
    this.pipe = this.pipe || await pipeline(
      "feature-extraction",
      this.modelName
    );

    const result = await this.pipe(text, POOLING_OPTIONS);
    return Array.from(result.data as Float32Array) as number[];
  }
}


const embedder = new Embedder();
export { embedder, TransformersJSEmbedding };

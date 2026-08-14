# LangChain Retrieval Agent

An example of a retrieval-augmented [LangChain](https://pinecone.io/learn/langchain) agent: it indexes the Stanford Question-Answering Dataset (SQuAD) into a [Pinecone](https://www.pinecone.io/) vector index, then answers questions by giving an OpenAI chat model a retriever tool over that index — so it can pull in relevant passages instead of relying on training data alone.

## Prerequisites

- Node.js 22+ and npm
- A [Pinecone](https://app.pinecone.io/) account and API key
- An [OpenAI](https://platform.openai.com/api-keys) API key

## Quickstart

Install dependencies:

```bash
npm install
```

Copy the environment template and fill in your keys:

```bash
cp .env.example .env
```

```sh
OPENAI_API_KEY=<your-openai-api-key>
PINECONE_API_KEY=<your-pinecone-api-key>
PINECONE_INDEX=langchain-retrieval-agent
```

`PINECONE_INDEX` is created for you (if it doesn't already exist) by the next step — it doesn't need to exist beforehand.

Build the index — downloads SQuAD, embeds ~19,000 deduplicated passages locally, and upserts them into Pinecone (this takes a few minutes):

```bash
npm run index
```

Ask the agent a question:

```bash
npm run chat
```

You should see something like:

```
Here are some interesting facts about the University of Notre Dame:

1. **Location and Name**: The University of Notre Dame du Lac, commonly known as Notre Dame, is
   located in South Bend, Indiana...
...
```

(The model's exact wording will vary between runs.)

## How it works

### Building the knowledge base (`npm run index`)

`squadLoader.ts` downloads the SQuAD JSON, flattens it into rows (one per question/passage pair) via `dataLoader.ts`, and deduplicates by passage text. `index.ts` calls it to load the full dataset into memory:

```typescript
import { loadSquad } from "./utils/squadLoader.js";

const squadData = await loadSquad();
```

Since the dataset is large, `index.ts` embeds and upserts it in batches via a generator that yields chunks of `Document`s:

```typescript
function* processInChunks(
  records: SquadRecord[],
  chunkSize: number
): Generator<Document[]> {
  for (let i = 0; i < records.length; i += chunkSize) {
    const chunk = records.slice(i, i + chunkSize);
    yield chunk.map(
      (record: SquadRecord) =>
        new Document({
          pageContent: record.context,
          metadata: {
            id: record["id"],
            question: record["question"],
            answer: record["answer"],
            context: record["context"],
          },
        })
    );
  }
}
```

Each chunk is embedded locally (via `@huggingface/transformers`, no external embedding API needed) and upserted into Pinecone:

```typescript
async function embedAndUpsert(records: SquadRecord[], chunkSize: number) {
  const chunkGenerator = processInChunks(records, chunkSize);
  const index = pinecone.index({ name: indexName }).namespace(NAMESPACE);

  for (const documents of chunkGenerator) {
    await embedder.embedBatch(
      documents,
      chunkSize,
      async (embeddings: PineconeRecord[]) => {
        await index.upsert({ records: embeddings });
        progressBar.increment(embeddings.length);
      }
    );
  }
}
```

`index.ts` ties this together: it creates the index if needed, initializes the embedder, and runs `embedAndUpsert` over the full dataset — this is the entry point for `npm run index`.

### Retrieval agent (`npm run chat`)

`chat.ts` opens a vector store on the same index, reading the passage text back from the `context` metadata field it was upserted under:

```typescript
import { PineconeStore } from "@langchain/pinecone";
import { getPineconeClient } from "utils/pinecone.ts";
import { TransformersJSEmbedding } from "embeddings.ts";

const indexName = getEnv("PINECONE_INDEX");
const pinecone = getPineconeClient();
const pineconeIndex = pinecone.index({ name: indexName });

const vectorStore = await PineconeStore.fromExistingIndex(
  new TransformersJSEmbedding({
    modelName: "Xenova/all-MiniLM-L6-v2",
  }),
  { pineconeIndex, namespace: "default", textKey: "context" },
);
```

It then wraps the vector store's retriever as a tool via `createRetrieverTool`:

```typescript
import { createRetrieverTool } from "@langchain/classic/tools/retriever";

const retriever = vectorStore.asRetriever(4);

const knowledgeBaseTool = createRetrieverTool(retriever, {
  name: "knowledge_base",
  description:
    "Search the knowledge base for background information when answering general knowledge questions about a topic.",
});
```

Finally, it combines an OpenAI chat model with the tool using `createAgent` — a LangGraph-based agent that wires the model to its tools and runs the reason/act loop internally:

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { createAgent } from "langchain";

const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
const agent = createAgent({ model, tools: [knowledgeBaseTool] });

const input = "can you tell me some facts about the University of Notre Dame?";
const result = await agent.invoke({ messages: [{ role: "user", content: input }] });

const finalMessage = result.messages.at(-1);
console.log(finalMessage?.content);
```

That's all for this example of building a retrieval-augmented conversational agent with OpenAI, Pinecone, and LangChain.

# LangChain Retrieval Agent

Chatbots can struggle with data freshness, knowledge about specific domains, or accessing internal documentation. By coupling agents with retrieval augmentation tools we no longer have these problems.

One the other side, using "naive" retrieval augmentation without the use of an agent means we will retrieve contexts with every query. Again, this isn't always ideal as not every query requires access to external knowledge.

Merging these methods gives us the best of both worlds. Let's see how that is done.

(See our [LangChain Handbook](https://pinecone.io/learn/langchain) for more on LangChain).

To begin, we must install the prerequisite libraries that we will be using in this applications.

To do so, simply run the following command:

```bash
npm install
```

## Configuration

This example needs both a Pinecone and an OpenAI API key. Copy the template file:

```sh
cp .env.example .env
```

And fill in your keys and index name:

```sh
OPENAI_API_KEY=<your-openai-api-key>
PINECONE_API_KEY=<your-pinecone-api-key>
PINECONE_INDEX="langchain-retrieval-agent"
```

## Importing the Libraries

We'll start by importing the necessary libraries. We'll be using the `@pinecone-database/pinecone` library to interact with Pinecone. We'll also be using the `danfojs-node` library to load the data into an easy to manipulate dataframe. We'll use the `Document` type from `@langchain/core` to keep the data structure consistent across the indexing process and retrieval agent.

We'll be using the `Embedder` class found in `embeddings.ts` to embed the data. We'll also be using the `cli-progress` library to display a progress bar.

To load the dataset used in the example, we'll be using a utility called `squadLoader.ts`.

```typescript
import { PineconeRecord } from "@pinecone-database/pinecone";
import { getEnv } from "utils/util.ts";
import { getPineconeClient } from "utils/pinecone.ts";
import cliProgress from "cli-progress";
import { Document } from "@langchain/core/documents";
import * as dfd from "danfojs-node";
import { embedder } from "embeddings.ts";
import { SquadRecord, loadSquad } from "./utils/squadLoader.js";
```

## Building the Knowledge Base

We start by constructing our knowledge base. We'll use a mostly prepared dataset called Stanford Question-Answering Dataset (SQuAD), downloaded directly from its host. The data will be loaded into a `Danfo` dataframe.

```typescript
const squadData = await loadSquad();
// Start the progress bar
progressBar.start(squadData.shape[0], 0);
```

Since the dataset could be pretty big, we'll use a generator function that will yield chunks of data to be processed.

```typescript
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
```

Next we'll create a function that will generate the embeddings and upsert them into Pinecone. We'll use the `processInChunks` generator function to process the data in chunks.

```typescript
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
```

Next, we'll set up the index, initialize the embedder and call `embedAndUpsert` to start the process. Run this with `npm run index`:

```typescript
const squadData = await loadSquad();
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
console.log(`Inserted ${squadData.shape[0]} documents into index ${indexName}`);
```

```sh
npm run index
```

The SQuAD dataset has around 19,000 unique passages after deduplication, so expect this to take a few minutes.

## Retrieval Agent

Now that we've built our index we can switch back over to LangChain. We start by initializing a vector store using the same index we just built, reading the passage text back from the `context` metadata field it was upserted under:

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

Next, we retrieve the most relevant passages for a query and expose them to the agent as a tool via `createRetrieverTool`:

```typescript
import { createRetrieverTool } from "@langchain/classic/tools/retriever";

const retriever = vectorStore.asRetriever(4);

const knowledgeBaseTool = createRetrieverTool(retriever, {
  name: "knowledge_base",
  description:
    "Search the knowledge base for background information when answering general knowledge questions about a topic.",
});
```

Finally, we combine an OpenAI chat model with the tool using `createAgent` — a LangGraph-based agent that wires the model to its tools and runs the reason/act loop internally:

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

Run this with `npm run chat`:

```sh
npm run chat
```

We should see something like this:

```
Here are some interesting facts about the University of Notre Dame:

1. **Location and Name**: The University of Notre Dame du Lac, commonly known as Notre Dame, is
   located in South Bend, Indiana...
...
```

(The model's exact wording will vary between runs.)

Looks great! That's all for this example of building a retrieval augmented conversational agent with OpenAI and Pinecone and LangChain.

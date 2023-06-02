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

## Importing the Libraries

We'll start by importing the necessary libraries. We'll be using the `@pinecone-database/pinecone` library to interact with Pinecone. We'll also be using the `danfojs-node` library to load the data into an easy to manipulate dataframe. We'll use the `Document` type from Langchain to keep the data structure consistent across the indexing process and retrieval agent.

We'll be using the `Embedder` class found in `embeddings.ts` to embed the data We'll also be using the `cli-progress` library to display a progress bar.

To load the dataset used in the example, we'll be using a utility called `squadLoader.js`.

```typescript
import { Vector, utils } from "@pinecone-database/pinecone";
import { getEnv } from "utils/util.ts";
import { getPineconeClient } from "utils/pinecone.ts";
import cliProgress from "cli-progress";
import { Document } from "langchain/document";
import * as dfd from "danfojs-node";
import { embedder } from "embeddings.ts";
import { SquadRecord, loadSquad } from "./utils/squadLoader.js";
```

## Building the Knowledge Base

We start by constructing our knowledge base. We'll use a mostly prepared dataset called Stanford Question-Answering Dataset (SQuAD) hosted on Hugging Face Datasets. We download using a simple data-loading utility library. The data will be loaded into a `Danfo` dataframe.

```typescript
const squadData = await loadSquad();
// Start the progress bar
progressBar.start(squadData.shape[0], 0);
```

Since the dataset could be pretty big, we'll use a generator function that will yield chunks of data to be processed.

```typescript
async function* processInChunks(
  dataFrame: dfd.DataFrame,
  chunkSize: number
): AsyncGenerator<Document[]> {
  for (let i = 0; i < dataFrame.shape[0]; i += chunkSize) {
    const chunk = await getChunk(dataFrame, i, chunkSize);
    const records = dfd.toJSON(chunk) as SquadRecord[];
    yield records.map(
      (record: SquadRecord) =>
        new Document({
          pageContent: record.context,
          metadata: {
            id: record["id"],
            question: record["question"],
            answer: record["answer"],
          },
        })
    );
  }
}
```

Next we'll create a funciton that will generate the embeddings and upsert them into Pinecone. We'll use the `processInChunks` generator function to process the data in chunks. We'll also use the `chunkedUpsert` method to insert the embeddings into Pinecone in batches.

```typescript
async function embedAndUpsert(dataFrame: dfd.DataFrame, chunkSize: number) {
  const chunkGenerator = processInChunks(dataFrame, chunkSize);
  const index = pineconeClient.Index(indexName);

  for await (const documents of chunkGenerator) {
    await embedder.embedBatch(
      documents,
      chunkSize,
      async (embeddings: Vector[]) => {
        await chunkedUpsert(index, embeddings, "default");
        progressBar.increment(embeddings.length);
      }
    );
  }
}
```

Next, we'll set up the index, initialize the embedder and call `embedAndUpsert` to start the process.

```typescript
try {
  const squadData = await loadSquad();
  await createIndexIfNotExists(pineconeClient, indexName, 384);

  progressBar.start(squadData.shape[0], 0);

  await embedder.init("Xenova/all-MiniLM-L6-v2");
  await embedAndUpsert(squadData, 100);

  progressBar.stop();
  console.log(
    `Inserted ${progressBar.getTotal()} documents into index ${indexName}`
  );
} catch (error) {
  console.error(error);
}
```

## Retrieval Agent

Now that we've build our index we can switch back over to LangChain. We start by initializing a vector store using the same index we just built. We do that like so:

```typescript
import { TransformersJSEmbedding } from "embeddings.ts";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { getPineconeClient } from "utils/pinecone.ts";

const indexName = getEnv("PINECONE_INDEX");

const pineconeClient = await getPineconeClient();
const pineconeIndex = pineconeClient.Index(indexName);

const vectorStore = await PineconeStore.fromExistingIndex(
  new TransformersJSEmbedding({
    modelName: "Xenova/all-MiniLM-L6-v2",
  }),
  { pineconeIndex }
);
```

We can use the `similaritySearch` method to do a pure semantic search (without the generation component).

```typescript
const result = await vectorStore.similaritySearch(
  "when was the college of engineering in the University of Notre Dame established?",
  3
);
console.log(result);
```

We should see the following results:

```
[
  Document {
    pageContent: 'The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study – aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering – with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively.',
    metadata: {
      answer: 'the 1870s',
      id: '5733a6424776f41900660f52',
      question: 'The College of Science began to offer civil engineering courses beginning at what time at Notre Dame?'
    }
  },
  Document {
    pageContent: 'The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study – aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering – with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively.',
    metadata: {
      answer: 'five',
      id: '5733a6424776f41900660f50',
      question: 'How many departments are within the Stinson-Remick Hall of Engineering?'
    }
  },
  Document {
    pageContent: 'The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study – aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering – with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively.',
    metadata: {
      answer: 'the College of Science',
      id: '5733a6424776f41900660f4f',
      question: 'Before the creation of the College of Engineering similar studies were carried out at which Notre Dame college?'
    }
  }
]
```

Looks like we're getting good results. Let's take a look at how we can begin integrating this into a conversational agent.

First, we'll create a `Vector Store` that will use the same embedding model as the one we used to build the index.

```typescript
const vectorStore = await PineconeStore.fromExistingIndex(
  new TransformersJSEmbedding({
    modelName: "Xenova/all-MiniLM-L6-v2",
  }),
  { pineconeIndex, namespace: "default", textKey: "context" }
);
```

Next, we'll initialize the tools used by the agent. Those will required a `model` (such as `OpenAI`) that will be responsible to generating the response. We'll combine the two by using a chain called `VectorDBQAChain`:

```typescript
const model = new OpenAI({});

const chain = VectorDBQAChain.fromLLM(model, vectorStore);

const kbTool = new ChainTool({
  name: "Knowledge Base",
  description:
    "use this tool when answering general knowledge queries to get more information about the topic",
  chain,
});
```

Finally, we'll create the agent executor that'll combine the model and the vector store tool:

```typescript
const executor = await initializeAgentExecutorWithOptions([kbTool], model, {
  agentType: "zero-shot-react-description",
});
```

Now we can use the executor to generate a response:

```typescript
const input = "can you tell me some facts about the University of Notre Dame?";
const result = await executor.call({ input });
console.log(`${result.output}`);
```

We should see something like this:

```
The University of Notre Dame has a First Year of Studies program established in 1962, provides academic advisors, and has been recognized by U.S. News & World Report as outstanding.
```

Looks great! We're also able to ask questions that refer to previous interactions in the conversation and the agent is able to refer to the conversation history to as a source of information.

That's all for this example of building a retrieval augmented conversational agent with OpenAI and Pinecone (the OP stack) and LangChain.

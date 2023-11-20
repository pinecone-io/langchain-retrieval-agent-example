# LangChain Retrieval Agent

Chatbots can struggle with data freshness, knowledge about specific domains, or accessing internal documentation. By coupling agents with retrieval augmentation tools we no longer have these problems.

One the other side, using "naive" retrieval augmentation without the use of an agent means we will retrieve contexts with every query. Again, this isn't always ideal as not every query requires access to external knowledge.

Merging these methods gives us the best of both worlds. Let's see how that is done.

(See our [LangChain Handbook](https://pinecone.io/learn/langchain) for more on LangChain).

## Setup

Prerequisites:
- `Node.js` version >=18.0.0

Clone the repository and install the prerequisite libraties that we will be using in this application.

```bash
git clone git@github.com:pinecone-io/langchain-retrieval-agent-example.git
cd langchain-retrieval-agent-example
npm install
```

### Configuration

In order to run this example, you have to supply the Pinecone credentials needed to interact with the Pinecone API. You can find these credentials in the Pinecone web console. This project uses `dotenv` to easily load values from the `.env` file into the environment when executing. 

Copy the template file:

```sh
cp .env.example .env
```

And fill in your API key and environment details:

```sh
OPENAI_API_KEY=<your-api-key>
PINECONE_API_KEY=<your-api-key>
PINECONE_ENVIRONMENT=<your-environment>
PINECONE_INDEX=langchain-retrieval-agent
```

`PINECONE_INDEX` is the name of the index where this demo will store and query embeddings. You can change `PINECONE_INDEX` to any name you like, but make sure the name not going to collide with any indexes you are already using.

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

```json
[
  {
    "answer": "a Marian place of prayer and reflection",
    "context": "The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study – aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering – with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively.",
    "id": "5733be284776f41900661181",
    "question": "What is the Grotto at Notre Dame?"
  },
  {
    "answer": "a golden statue of the Virgin Mary",
    "context": "All of Notre Dame's undergraduate students are a part of one of the five undergraduate colleges at the school or are in the First Year of Studies program. The First Year of Studies program was established in 1962 to guide incoming freshmen in their first year at the school before they have declared a major. Each student is given an academic advisor from the program who helps them to choose classes that give them exposure to any major in which they are interested. The program also includes a Learning Resource Center which provides time management, collaborative learning, and subject tutoring. This program has been recognized previously, by U.S. News & World Report, as outstanding.",
    "id": "5733be284776f4190066117e",
    "question": "What sits on top of the Main Building at Notre Dame?"
  },
  {
    "answer": "the 1870s",
    "context": "In 1919 Father James Burns became president of Notre Dame, and in three years he produced an academic revolution that brought the school up to national standards by adopting the elective system and moving away from the university's traditional scholastic and classical emphasis. By contrast, the Jesuit colleges, bastions of academic conservatism, were reluctant to move to a system of electives. Their graduates were shut out of Harvard Law School for that reason. Notre Dame continued to grow over the years, adding more colleges, programs, and sports teams. By 1921, with the addition of the College of Commerce, Notre Dame had grown from a small college to a university with five colleges and a professional law school. The university continued to expand and add new residence halls and buildings with each subsequent president.",
    "id": "5733a6424776f41900660f52",
    "question": "The College of Science began to offer civil engineering courses beginning at what time at Notre Dame?"
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
The University of Notre Dame is a Catholic research university located in South Bend, Indiana, United States. It is consistently ranked among the top twenty universities in the United States. It has four colleges (Arts and Letters, Science, Engineering, Business) and an Architecture School. Its graduate program has more than 50 master's, doctoral and professional degree programs. It also has a First Year of Studies program and an Office of Sustainability. Father Gustavo Gutierrez, the founder of Liberation Theology is a current faculty member.
```

Looks great! We're also able to ask questions that refer to previous interactions in the conversation and the agent is able to refer to the conversation history to as a source of information.

That's all for this example of building a retrieval augmented conversational agent with OpenAI and Pinecone (the OP stack) and LangChain.

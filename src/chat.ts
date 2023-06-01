import { PineconeStore } from "langchain/vectorstores/pinecone";
import { OpenAI } from "langchain/llms/openai";
import { VectorDBQAChain } from "langchain/chains";
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { ChainTool } from "langchain/tools";
import { getPineconeClient } from "utils/pinecone.ts";
import { getEnv } from "utils/util.ts";
import { TransformersJSEmbedding } from "embeddings.ts";

const indexName = getEnv("PINECONE_INDEX");

const pineconeClient = await getPineconeClient();
const pineconeIndex = pineconeClient.Index(indexName);

const vectorStore = await PineconeStore.fromExistingIndex(
  new TransformersJSEmbedding({
    modelName: "Xenova/all-MiniLM-L6-v2"
  }),
  { pineconeIndex, namespace: "default", textKey: "context" },
);

const result = await vectorStore.similaritySearch("when was the college of engineering in the University of Notre Dame established?", 3);
// // console.log(result);

console.log(result);

// const embedder = new TransformersJSEmbedding({
//   modelName: "Xenova/all-MiniLM-L6-v2"
// });

// const embedding = await embedder.embedQuery("when was the college of engineering in the University of Notre Dame established?");

// const queryRequest: QueryRequest = {
//   vector: embedding,
//   topK: 3,
//   namespace: "default",
//   includeMetadata: true,
// };

// const result2 = await pineconeIndex.query({ queryRequest });
// console.log(result2.matches?.map(x => x.metadata));

// const model = new OpenAI({});

// const chain = VectorDBQAChain.fromLLM(model, vectorStore);


// const kbTool = new ChainTool({
//   name: "Knowledge Base",
//   description:
//     "use this tool when answering general knowledge queries to get more information about the topic",
//   chain,
// });

// const executor = await initializeAgentExecutorWithOptions([kbTool], model, {
//   agentType: "zero-shot-react-description",
//   verbose: false,
// });
// console.log("Loaded agent.");

// const input = "can you tell me some facts about the University of Notre Dame?";

// console.log(`Executing with input "${input}"...`);

// const result = await executor.call({ input });

// console.log(`${result.output}`);
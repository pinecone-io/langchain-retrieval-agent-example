import { PineconeStore } from "@langchain/pinecone";
import { ChatOpenAI } from "@langchain/openai";
import { createAgent } from "langchain";
import { createRetrieverTool } from "@langchain/classic/tools/retriever";
import { getPineconeClient } from "utils/pinecone.ts";
import { getEnv } from "utils/util.ts";
import { TransformersJSEmbedding } from "embeddings.ts";

const indexName = getEnv("PINECONE_INDEX");

const pinecone = getPineconeClient();
const pineconeIndex = pinecone.index({ name: indexName });

// The passage text was upserted into the `context` metadata field, so the
// vector store reads it back from there.
const vectorStore = await PineconeStore.fromExistingIndex(
  new TransformersJSEmbedding({
    modelName: "Xenova/all-MiniLM-L6-v2"
  }),
  { pineconeIndex, namespace: "default", textKey: "context" },
);

// Retrieve the 4 most relevant passages for a query and expose them as a tool.
const retriever = vectorStore.asRetriever(4);

const knowledgeBaseTool = createRetrieverTool(retriever, {
  name: "knowledge_base",
  description:
    "Search the knowledge base for background information when answering general knowledge questions about a topic.",
});

const model = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

// v1 agents are LangGraph-based; createAgent wires the chat model to the tools
// and runs the reason/act loop internally (replacing the removed
// initializeAgentExecutorWithOptions).
const agent = createAgent({ model, tools: [knowledgeBaseTool] });
console.log("Loaded agent.");

const input = "can you tell me some facts about the University of Notre Dame?";

console.log(`Executing with input "${input}"...`);

const result = await agent.invoke({ messages: [{ role: "user", content: input }] });

// The final answer is the content of the last message the agent produced. It is
// usually a plain string, but can be an array of content blocks.
const finalMessage = result.messages.at(-1);
const answer =
  typeof finalMessage?.content === "string"
    ? finalMessage.content
    : (finalMessage?.content ?? [])
        .map((part) => ("text" in part ? part.text : ""))
        .join("");

console.log(answer);

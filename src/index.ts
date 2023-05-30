import * as dotenv from "dotenv";
import { utils } from '@pinecone-database/pinecone';
import { getEnv } from "utils/util.ts";
import { getPineconeClient } from "utils/pinecone.ts";
import { Document } from 'langchain/document';
import * as dfd from "danfojs-node";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { OpenAI } from "langchain/llms/openai";
import { RetrievalQAChain } from "langchain/chains";

import { loadSquad } from "./utils/squadLoader.js";

const { createIndexIfNotExists } = utils;

dotenv.config();

const indexName = getEnv("PINECONE_INDEX");

const pineconeClient = await getPineconeClient();

await createIndexIfNotExists(pineconeClient, indexName, 1536);

const pineconeIndex = pineconeClient.Index(indexName);

const squadData = await loadSquad();

const records = dfd.toJSON(squadData.head()) as any[];

const documents = records.map((record) => {
  const document = new Document({
    pageContent: record.context,
    metadata: {
      id: record["qas.id"],
      question: record["qas.question"],
      answer: record["qas.answers.text"],
    },
  });

  return document;
});

console.log(documents);

await PineconeStore.fromDocuments(documents, new OpenAIEmbeddings(), {
  pineconeIndex,
});

const vectorStore = await PineconeStore.fromExistingIndex(
  new OpenAIEmbeddings(),
  { pineconeIndex }
);
const model = new OpenAI({});

const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
const res = await chain.call({
  query: "What is the Grotto at Notre Dame?",
});
console.log({ res });

// const documents = records.map((record) => ({});


// await embedder.init();
// const embedded = await embedder.embed("Hello world!");
// console.log(embedded);

// const model = new OpenAI({
//   modelName: "gpt-3.5-turbo",
//   openAIApiKey: process.env.OPENAI_API_KEY,
// });

// const res = await model.call(
//   "What's a good idea for an application to build with GPT-3?"
// );

// console.log(res);



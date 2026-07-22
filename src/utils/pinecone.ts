import { Pinecone } from "@pinecone-database/pinecone";
import { config } from "dotenv";
import { getEnv, validateEnvironmentVariables } from "./util.js";

config();

let pinecone: Pinecone | null = null;

// Returns a memoized Pinecone client. The v8 (serverless) SDK constructs
// synchronously from just an API key — there is no `.init()` and no
// `environment`; a single key manages every region.
export const getPineconeClient = (): Pinecone => {
  validateEnvironmentVariables();

  if (pinecone) {
    return pinecone;
  }
  pinecone = new Pinecone({
    apiKey: getEnv("PINECONE_API_KEY"),
    sourceTag: "pinecone:langchain_retrieval_agent_example",
  });

  return pinecone;
};

import { PineconeClient } from "@pinecone-database/pinecone";
import { config } from "dotenv";
import { getEnv, validateEnvironmentVariables } from "./util.js";

config();

let pineconeClient: PineconeClient | null = null;

// Returns a PineconeClient instance
export const getPineconeClient: () => Promise<PineconeClient> = async () => {
  validateEnvironmentVariables();

  if (pineconeClient) {
    return pineconeClient;
  }
  pineconeClient = new PineconeClient();

  await pineconeClient.init({
    apiKey: getEnv("PINECONE_API_KEY"),
    environment: getEnv("PINECONE_ENVIRONMENT"),
  });

  return pineconeClient;
};